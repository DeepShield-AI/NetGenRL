import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from dataset import CustomDataset
from util import *
from rollout import Rollout
from GANloss import GANLoss
import time

# def seq_to_wordvec(seqs, wv):
#     seqs = seqs.cpu().numpy()
#     seqs = seqs.tolist()
    
#     seqs_wv = [[[wv[SEQ_DICT[i]].wv[str(value)] for i,value in enumerate(item)] for item in seq] for seq in seqs]
#     return torch.tensor(seqs_wv,dtype=torch.float32)

def get_non_matching_labels_one_hot(labels_indices, num_classes, device):
    # 获取当前批次的标签
    current_labels = labels_indices.clone()

    random_labels = torch.randint(0, num_classes, (labels_indices.size(0),), device=device)

    # 找到与 labels 中相同位置的索引
    same_positions = random_labels == current_labels

    # 对这些相同位置的数值进行修改
    while torch.any(same_positions):  # 如果有位置相同，继续修改
        random_labels[same_positions] = torch.randint(0, num_classes, (same_positions.sum(),), device=device)  # 修改相同位置的数值
        same_positions = random_labels == current_labels  # 更新相同位置
    
    # 将随机标签转换为 one-hot 编码
    wrong_labels_one_hot = torch.zeros(labels_indices.size(0), num_classes, device=device)
    wrong_labels_one_hot.scatter_(1, random_labels.view(-1, 1), 1)

    return wrong_labels_one_hot


def compute_gradient_penalty(discriminator, real_seqs, fake_seqs, labels, lengths, device,lambda_gp=10.0):
    """
    计算梯度惩罚以确保Lipschitz连续性。
    :param discriminator: 判别器模型
    :param real_images: 真实图像 batch
    :param fake_images: 生成图像 batch
    :param real_metadata: 真实图像对应的元数据 batch
    :param fake_metadata: 生成图像对应的元数据 batch
    :param device: 当前设备
    :return: 梯度惩罚值
    """
    batch_size = real_seqs.size(0)
    
    # 随机插值生成新的样本
    alpha = torch.rand(batch_size, 1, 1).to(device)  # 随机采样权重
    alpha = alpha.expand_as(real_seqs)
    interpolated_seqs = alpha * real_seqs + (1 - alpha) * fake_seqs
    interpolated_seqs.requires_grad_(True)
    # interpolated_metadata = alpha * real_metadata + (1 - alpha) * fake_metadata
    
    # 将插值图像和元数据合并
    # interpolated_input = (interpolated_images, interpolated_metadata)

    # 计算插值样本的判别器输出
    interpolated_scores = discriminator.forward(labels,interpolated_seqs,lengths)  # 注意这里要传入图像和元数据

    # 计算梯度
    gradients = torch.autograd.grad(outputs=interpolated_scores, inputs=interpolated_seqs,
                                   grad_outputs=torch.ones_like(interpolated_scores),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # 计算梯度的L2范数
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # 计算梯度惩罚
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()  # L2范数距离1的平方
    return gradient_penalty*lambda_gp

def pre_train(generator, discriminator, dataloader, generator_epoch, discirminator_epoch, device):
    # Pre-train generator
    # gen_criterion = F.nll_loss(reduction='none')
    gen_optimizer = optim.Adam(generator.parameters(),lr=0.0001, betas=(0.5, 0.999))
    x_list = generator.x_list
    
    for epoch in range(generator_epoch):
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seq_len = seqs.size(1)
            seq_dim = seqs.size(2)
            
            seqs = seqs.to(device) # (batch_size, seq_len, seq_dim)
            labels = labels.to(device)
            weights = weights.unsqueeze(-1).to(device)
            lengths = lengths.to(device)
            mask = torch.arange(seq_len).unsqueeze(0).to(device) < lengths.unsqueeze(1)
            zero = torch.zeros(batch_size, 1, seq_dim).long().to(device)
            
            seqs_seed = torch.cat([zero, seqs[:,:-1,:]], dim=1) # (batch_size, seq_len, seq_dim)
            
            fake_preds = generator.forward(labels, lengths, seqs_seed, seqs) # (batch_size, seq_len, prob_dim)
            
            # target = torch.cat([seqs[:,:-1,:], zero], dim=1) # (batch_size, seq_len, seq_dim)
            target = seqs # (batch_size, seq_len, seq_dim)
            
            count = 0
            
            g_loss = 0.0
            for j, x_len in enumerate(x_list):
                # print(fake_preds[:,:,count:count+x_len].shape, target[:,:,j].shape)
                loss = F.nll_loss(fake_preds[:,:,count:count+x_len].view(-1,x_len), target[:,:,j].view(-1),reduction='none')
                loss = loss.view(batch_size, seq_len)  # 恢复为 (batch_size, seq_len)
                loss = loss * mask * weights
                g_loss += loss.sum()
                count += x_len
                
            g_loss = g_loss/len(x_list)
            gen_optimizer.zero_grad()
            g_loss.backward()
            gen_optimizer.step()
            
        print('Pre-train Epoch [%d] Generator Loss: %f'% (epoch, g_loss))
        
        torch.save(generator.state_dict(), '../save_seq/generator_pre.pth')
        
    # dis_criterion = F.nll_loss(reduction='none')
    # dis_optimizer = optim.Adam(discriminator.parameters())
    # Pre-train discriminator
    
    # dis_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.00005)
    # dis_optimizer = optim.Adam(discriminator.parameters(),lr=0.0002, betas=(0.5, 0.999))
    dis_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)
    # dis_criterion = nn.NLLLoss(reduction='sum')

    for epoch in range(discirminator_epoch):
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seq_len = seqs.size(1)
            seq_dim = seqs.size(2)
            
            seqs = seqs.to(device) # (batch_size, seq_len, seq_dim)
            labels = labels.to(device)
            weights = weights.unsqueeze(-1).to(device)
            # lengths = lengths.to(device)
            
            fake_seqs = generator.sample(batch_size, labels, lengths) 
            
            fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
            real_seqs_wv = discriminator.seq2wv(seqs).to(device)
            # print(fake_seqs.shape)
            fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
            real_validity = discriminator.forward(labels, real_seqs_wv, lengths) * weights
            
            # real_loss = F.binary_cross_entropy_with_logits(real_validity, torch.ones_like(real_validity))  # 真实数据目标是 1
            # fake_loss = F.binary_cross_entropy_with_logits(fake_validity, torch.zeros_like(fake_validity)) 
            
            # zero = torch.zeros(batch_size).long().to(device)
            # one = torch.ones(batch_size).long().to(device)
            
            # real_loss = dis_criterion(real_validity, one)
            # fake_loss = dis_criterion(fake_validity, zero)
            
            dis_optimizer.zero_grad()
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            # d_loss = real_loss + fake_loss
            # d_loss = math.exp(real_loss + fake_loss)
            
            with torch.backends.cudnn.flags(enabled=False):
                gp = compute_gradient_penalty(discriminator, real_seqs_wv, fake_seqs_wv,labels, lengths,device)
            total_d_loss = d_loss + gp
            total_d_loss.backward()
            # dis_optimizer.step()
            # d_loss.backward()
            dis_optimizer.step()
            
        print('Pre-train Epoch [%d] Discriminator Loss: %f'% (epoch, d_loss))
        
        torch.save(discriminator.state_dict(), '../save_seq/discriminator_pre.pth')
            
            
    

# %%
def train(generator, discriminator, dataloader, epochs, device, clip_value=0.01, alpha = 1.0):
    # 使用Wasserstein GAN损失
    optimizer_g = optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=0.00005)
    
    # optimizer_g = optim.Adam(generator.parameters(),lr=0.0002, betas=(0.5, 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(),lr=0.0002, betas=(0.5, 0.999))
    
    rollout = Rollout(generator, 0.8)
    gan_loss = GANLoss(generator.x_list)
    dis_criterion = nn.NLLLoss(reduction='sum')

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seqs = seqs.to(device)
            labels = labels.to(device)
            weights = weights.unsqueeze(-1).to(device)
            # lengths = lengths.to(device)
            # metadatas = metadatas.to(device)
            
            sample_time = 0
            reward_time = 0
            g_forward_time = 0
            l_forward_time = 0
            d_forward_time = 0
            grad_time = 0
            other_time = 0

            # 创建标签
            # valid = torch.ones(batch_size, 1).to(device)   # 真实数据+对应标签 -> 1
            # fake = torch.zeros(batch_size, 1).to(device)   # 生成数据+标签 -> 0
            
            start_time = time.perf_counter()
            samples = generator.sample(batch_size, labels, lengths) # (batch_size, seq_len, seq_dim)
            zeros = torch.zeros((batch_size, 1, seq_dim)).type(torch.LongTensor).to(device)
            inputs = torch.cat([zeros, samples], dim=1)[:,:-1,:].contiguous()
            targets = samples.contiguous().view(-1,seq_dim)
            sample_time += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            rewards = rollout.get_reward(samples, N_ROLL, discriminator, labels, lengths)
            # rewards_exp = torch.exp(rewards.clone()).contiguous().view((-1,)).to(device)
            rewards_exp = rewards.clone().contiguous().view((-1,)).to(device)
            reward_time += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            prob = generator.forward(labels, lengths, inputs, samples) # (batch_size, seq_len, prob_dim)
            g_forward_time += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            g_loss = gan_loss.forward(prob, targets, rewards_exp, device, weights)
            l_forward_time += time.perf_counter() - start_time
            # print(g_loss)
            
            start_time = time.perf_counter()
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            rollout.update_params()
            other_time += time.perf_counter() - start_time

            # 训练判别器
            # optimizer_md.zero_grad()
            for _ in range(N_CRITIC):
                optimizer_d.zero_grad()
                # real_validity = discriminator(labels, seqs, lengths)
                #noise = torch.randn(batch_size, generator.noise_dim).to(device)
                start_time = time.perf_counter()
                fake_seqs = generator.sample(batch_size, labels, lengths)
                sample_time += time.perf_counter() - start_time
                
                start_time = time.perf_counter()
                fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
                real_seqs_wv = discriminator.seq2wv(seqs).to(device)
                # print(fake_seqs.shape)
                fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
                real_validity = discriminator.forward(labels, real_seqs_wv, lengths) * weights
                d_forward_time += time.perf_counter() - start_time
                # fake_validity = discriminator(labels, fake_seqs.detach(),lengths)
                
                
                # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) 
                
                
            
                d_loss_real = -torch.mean(real_validity)
                d_loss_fake = torch.mean(fake_validity)
                
                # d_loss_real = F.binary_cross_entropy_with_logits(real_validity, torch.ones_like(real_validity))  # 真实数据目标是 1
                # d_loss_fake = F.binary_cross_entropy_with_logits(fake_validity, torch.zeros_like(fake_validity)) 
                
                # zero = torch.zeros(batch_size).long().to(device)
                # one = torch.ones(batch_size).long().to(device)
            
                # d_loss_real = dis_criterion(real_validity, one)
                # d_loss_fake = dis_criterion(fake_validity, zero)
                
                d_loss = d_loss_real + d_loss_fake
            
                # dis_optimizer.zero_grad()
                
                start_time = time.perf_counter()
                with torch.backends.cudnn.flags(enabled=False):
                    # gp = compute_gradient_penalty(discriminator, seqs, fake_seqs.detach(),labels, lengths,device)
                    gp = compute_gradient_penalty(discriminator, real_seqs_wv, fake_seqs_wv,labels, lengths,device)
                grad_time += time.perf_counter() - start_time
                
                start_time = time.perf_counter()
                total_d_loss = d_loss + gp
                total_d_loss.backward()
                
                # d_loss.backward()
                optimizer_d.step()
                other_time += time.perf_counter() - start_time
                
            # print("sample_time: ", sample_time, "reward_time: ", reward_time, "g_forward_time: ", g_forward_time, "l_forward_time: ", l_forward_time, "d_forward_time: ", d_forward_time, "grad_time: ", grad_time, "other_time: ", other_time)
                

        # 每个epoch结束时保存模型
        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()} ({d_loss_real.item()}, {d_loss_fake.item()}), G Loss: {g_loss.item()}.")
        # print(f"Epoch [{epoch+1}/{epochs}], Real: {rp}, Fake: {fp}.")

        torch.save(generator.state_dict(), '../save_seq/generator.pth')
        torch.save(discriminator.state_dict(), '../save_seq/discriminator.pth')

        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), f'../save_seq/generator_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'../save_seq/discriminator_{epoch+1}.pth')
            
        torch.cuda.empty_cache()


# %%
# def generate_data(generator, label, noise, device):
#     # 设置模型为评估模式
#     generator.eval()

#     # 生成图像和元数据
#     generated_images, metadata = generator(label, noise)
    
#     return generated_images, metadata

# %%
if __name__ == '__main__':
    label_dim = len(LABEL_DICT) 
    # image_dim = (1, NPRINT_REAL_WIDTH, NPRINT_REAL_WIDTH)  # 生成单通道图像
    # metadata_dim = 2  # 元数据维度为2
    # noise_dim = 100  # 噪声维度
    batch_size = 64
    epochs = 1500
    seq_dim = SEQ_DIM
    max_seq_len = MAX_SEQ_LEN
    # x_list = [MAX_TIME, MAX_PKT_LEN]
    
    data_folder = '../data/' + DATASET + '/'
    bins_file_name = '../bins/bins_' + DATASET + '.json'
    wordvec_file_name = '../wordvec/word_vec_' + DATASET + '.json'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(wordvec_file_name, 'r') as f:
        wv_dict = json.load(f)
    
    wv = {}
    for key, metrics in wv_dict.items():
        wv[key] = torch.tensor(metrics, dtype=torch.float32).to(device)

    # print(len(wv['time'].wv.key_to_index),len(wv['pkt_len'].wv.key_to_index))
    x_list = [wv_tensor.size(0) for wv_tensor in wv.values()]
    # print(x_list)
   

    # 创建生成器、判别器和cGAN
    # generator = Generator(label_dim, noise_dim, image_dim, metadata_dim)
    # discriminator = Discriminator(label_dim, image_dim, metadata_dim)
    # cgan = ConditionalGAN(generator, discriminator)

    

    
    # device = "cpu"
    
    # cg = ConditionalGAN(label_dim,noise_dim,seq_dim,max_seq_len,device)
    generator = Generator(label_dim,seq_dim,max_seq_len,x_list,device)
    discriminator = Discriminator(label_dim,WORD_VEC_SIZE * len(SERY_LIST) + META_VEC_SIZE * len(META_LIST) ,max_seq_len,x_list,wv,device)
    
    generator.to(device)
    discriminator.to(device)
    
    # for wv_tensor in wv.values():
    #     wv_tensor = wv_tensor.to(device)

    print(device)

    print("Process dataset...")

    dataset = CustomDataset(data_folder = data_folder, bins_file=bins_file_name, class_mapping=LABEL_DICT,max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # checkpoint_g = torch.load('../save_seq/generator.pth')  # 加载保存的权重字典
    # generator.load_state_dict(checkpoint_g) 

    # checkpoint_d = torch.load('../save_seq/discriminator.pth')  # 加载保存的权重字典
    # discriminator.load_state_dict(checkpoint_d) 
    
    print("Pre-training...")
    pre_train(generator, discriminator, dataloader, 5, 5, device)

    print("Trainning...")
    train(generator, discriminator, dataloader, epochs, device)