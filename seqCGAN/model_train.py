import torch
import torch.optim as optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader
from torch import nn
from seqCGAN.generator import Generator
from seqCGAN.discriminator import Discriminator
from seqCGAN.dataset import CustomDataset
from seqCGAN.rollout import Rollout
from seqCGAN.GANloss import GANLoss
from seqCGAN.valueNet import ValueNet
import time
from post_process.check_bestmodel import get_real_data, SequenceDataset
import random
from seqCGAN.evaluation import *


def get_non_matching_labels_one_hot(labels_indices, num_classes, device):
    current_labels = labels_indices.clone()

    random_labels = torch.randint(0, num_classes, (labels_indices.size(0),), device=device)
    same_positions = random_labels == current_labels
    while torch.any(same_positions): 
        random_labels[same_positions] = torch.randint(0, num_classes, (same_positions.sum(),), device=device)  # 修改相同位置的数值
        same_positions = random_labels == current_labels 

    wrong_labels_one_hot = torch.zeros(labels_indices.size(0), num_classes, device=device)
    wrong_labels_one_hot.scatter_(1, random_labels.view(-1, 1), 1)

    return wrong_labels_one_hot


def compute_gradient_penalty(discriminator, real_seqs, fake_seqs, labels, lengths, device,lambda_gp=10.0):
    batch_size = real_seqs.size(0)
    
    alpha = torch.rand(batch_size, 1, 1).to(device) 
    alpha = alpha.expand_as(real_seqs)
    interpolated_seqs = alpha * real_seqs + (1 - alpha) * fake_seqs
    interpolated_seqs.requires_grad_(True)
    interpolated_scores = discriminator.forward(labels,interpolated_seqs,lengths) 
    interpolated_scores_mean = discriminator.cal_value(lengths,interpolated_scores) 

    # gradients = torch.autograd.grad(outputs=interpolated_scores, inputs=interpolated_seqs,
    #                                grad_outputs=torch.ones_like(interpolated_scores),
    #                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = torch.autograd.grad(outputs=interpolated_scores_mean, inputs=interpolated_seqs,
                                   grad_outputs=torch.ones_like(interpolated_scores_mean),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    

    gradient_penalty = ((gradient_norm - 1) ** 2).mean() 
    return gradient_penalty*lambda_gp

def pre_train(generator, discriminator, dataloader, generator_epoch, discirminator_epoch, device, model_path):
    # Pre-train generator
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
            
            target = seqs # (batch_size, seq_len, seq_dim)
            
            count = 0
            
            g_loss = 0.0
            for j, x_len in enumerate(x_list):
                loss = F.nll_loss(fake_preds[:,:,count:count+x_len].view(-1,x_len), target[:,:,j].view(-1),reduction='none')
                loss = loss.view(batch_size, seq_len)  # (batch_size, seq_len)
                loss = loss * mask * weights
                g_loss += loss.sum()
                count += x_len
                
            g_loss = g_loss/len(x_list)
            gen_optimizer.zero_grad()
            g_loss.backward()
            gen_optimizer.step()
            
        print('Pre-train Epoch [%d] Generator Loss: %f'% (epoch, g_loss))
        
        torch.save(generator.state_dict(), f'{model_path}generator_pre.pth')

    # Pre-train discriminator
    dis_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    for epoch in range(discirminator_epoch):
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seq_len = seqs.size(1)
            seq_dim = seqs.size(2)
            
            seqs = seqs.to(device) # (batch_size, seq_len, seq_dim)
            labels = labels.to(device)
            weights = weights.unsqueeze(-1).to(device)
            
            fake_seqs = generator.sample(batch_size, labels, lengths) 
            
            fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
            real_seqs_wv = discriminator.seq2wv(seqs).to(device)
            fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
            real_validity = discriminator.forward(labels, real_seqs_wv, lengths) * weights
            
            dis_optimizer.zero_grad()
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            
            with torch.backends.cudnn.flags(enabled=False):
                gp = compute_gradient_penalty(discriminator, real_seqs_wv, fake_seqs_wv,labels, lengths,device)
            total_d_loss = d_loss + gp
            total_d_loss.backward()
            dis_optimizer.step()
            
        print('Pre-train Epoch [%d] Discriminator Loss: %f'% (epoch, d_loss))
        
        torch.save(discriminator.state_dict(), f'{model_path}discriminator_pre.pth')
            
            
    

# %%
def train(generator, discriminator, dataloader, epochs, device, seq_dim, n_roll, n_critic, model_path, checkpoint, real_data, bins_data, meta_attrs, sery_attrs, label_dict):

    optimizer_g = optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=0.00005)
    
    rollout = Rollout(generator, 0.8)
    gan_loss = GANLoss(generator.x_list)

    for epoch in range(epochs):
        start = time.perf_counter()
        torch.cuda.empty_cache()
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seqs = seqs.to(device)
            labels = labels.to(device)
            weights = weights.unsqueeze(-1).to(device)
            
            sample_time = 0
            reward_time = 0
            g_forward_time = 0
            l_forward_time = 0
            d_forward_time = 0
            grad_time = 0
            other_time = 0
            
            start_time = time.perf_counter()
            samples = generator.sample(batch_size, labels, lengths) # (batch_size, seq_len, seq_dim)
            zeros = torch.zeros((batch_size, 1, seq_dim)).type(torch.LongTensor).to(device)
            inputs = torch.cat([zeros, samples], dim=1)[:,:-1,:].contiguous()
            targets = samples.contiguous().view(-1,seq_dim)
            sample_time += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            rewards = rollout.get_reward(samples, n_roll, discriminator, labels, lengths)
            rewards_exp = rewards.clone().contiguous().view((-1,)).to(device)
            reward_time += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            prob = generator.forward(labels, lengths, inputs, samples) # (batch_size, seq_len, prob_dim)
            g_forward_time += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            g_loss = gan_loss.forward(prob, targets, rewards_exp, device, weights)
            l_forward_time += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            rollout.update_params()
            other_time += time.perf_counter() - start_time

            # 训练判别器
            for _ in range(n_critic):
                optimizer_d.zero_grad()
                start_time = time.perf_counter()
                fake_seqs = generator.sample(batch_size, labels, lengths)
                sample_time += time.perf_counter() - start_time
                
                start_time = time.perf_counter()
                fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
                real_seqs_wv = discriminator.seq2wv(seqs).to(device)
                fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
                real_validity = discriminator.forward(labels, real_seqs_wv, lengths) * weights
                d_forward_time += time.perf_counter() - start_time
                
            
                d_loss_real = -torch.mean(real_validity)
                d_loss_fake = torch.mean(fake_validity)
                
                d_loss = d_loss_real + d_loss_fake

                
                start_time = time.perf_counter()
                with torch.backends.cudnn.flags(enabled=False):
                    gp = compute_gradient_penalty(discriminator, real_seqs_wv, fake_seqs_wv,labels, lengths,device)
                grad_time += time.perf_counter() - start_time
                
                start_time = time.perf_counter()
                total_d_loss = d_loss + gp
                total_d_loss.backward()
                
                optimizer_d.step()
                other_time += time.perf_counter() - start_time
                
            # print("sample_time: ", sample_time, "reward_time: ", reward_time, "g_forward_time: ", g_forward_time, "l_forward_time: ", l_forward_time, "d_forward_time: ", d_forward_time, "grad_time: ", grad_time, "other_time: ", other_time)
        end = time.perf_counter()
        
        
        fake_data = get_fake_json(label_dict, len(label_dict), real_data, generator, bins_data, sery_attrs, meta_attrs, batch_size)
        eu_sum = cal_euler(real_data, fake_data, label_dict)
        hm_sum = cal_hamming(real_data, fake_data, label_dict)

        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()} ({d_loss_real.item()}, {d_loss_fake.item()}), G Loss: {g_loss.item()}, time cost: {end - start}, Euler distance: {eu_sum}, Hamming distance: {hm_sum}.")     

        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()} ({d_loss_real.item()}, {d_loss_fake.item()}), G Loss: {g_loss.item()}.")

        torch.save(generator.state_dict(), f'{model_path}generator.pth')
        torch.save(discriminator.state_dict(), f'{model_path}discriminator.pth')

        if (epoch + 1) % checkpoint == 0:
            torch.save(generator.state_dict(), f'{model_path}generator_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'{model_path}discriminator_{epoch+1}.pth')
            
        torch.cuda.empty_cache()
        
def step_pre_train(generator, discriminator, value_net, dataloader, generator_epoch, discirminator_epoch, value_net_epoch, device, model_path):
    # Pre-train generator
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
            
            target = seqs # (batch_size, seq_len, seq_dim)
            
            count = 0
            
            g_loss = 0.0
            for j, x_len in enumerate(x_list):
                loss = F.nll_loss(fake_preds[:,:,count:count+x_len].view(-1,x_len), target[:,:,j].view(-1),reduction='none')
                loss = loss.view(batch_size, seq_len)  # (batch_size, seq_len)
                loss = loss * mask * weights
                g_loss += loss.sum()
                count += x_len
                
            g_loss = g_loss/len(x_list)
            gen_optimizer.zero_grad()
            g_loss.backward()
            gen_optimizer.step()
            
        print('Pre-train Epoch [%d] Generator Loss: %f'% (epoch, g_loss))
        
        torch.save(generator.state_dict(), f'{model_path}generator_pre.pth')

    # Pre-train discriminator
    dis_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    for epoch in range(discirminator_epoch):
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seq_len = seqs.size(1)
            seq_dim = seqs.size(2)
            
            seqs = seqs.to(device) # (batch_size, seq_len, seq_dim)
            labels = labels.to(device)
            weights = weights.unsqueeze(-1).to(device)
            
            fake_seqs = generator.sample(batch_size, labels, lengths) 
            
            fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
            real_seqs_wv = discriminator.seq2wv(seqs).to(device)
            # fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
            # real_validity = discriminator.forward(labels, real_seqs_wv, lengths) * weights
            
            fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths)
            real_validity = discriminator.forward(labels, real_seqs_wv, lengths)
            
            fake_validity_mean = discriminator.cal_value(lengths,fake_validity) * weights
            real_validity_mean = discriminator.cal_value(lengths,real_validity) * weights
            
            dis_optimizer.zero_grad()
            
            # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            d_loss = -torch.mean(real_validity_mean) + torch.mean(fake_validity_mean)
            
            with torch.backends.cudnn.flags(enabled=False):
                gp = compute_gradient_penalty(discriminator, real_seqs_wv, fake_seqs_wv,labels, lengths,device)
            total_d_loss = d_loss + gp
            total_d_loss.backward()
            dis_optimizer.step()
            
        print('Pre-train Epoch [%d] Discriminator Loss: %f'% (epoch, d_loss))
        
        torch.save(discriminator.state_dict(), f'{model_path}discriminator_pre.pth')
        
    # Pre-train value net
    val_optimizer = optim.Adam(generator.parameters(),lr=0.0001, betas=(0.5, 0.999))
    mse_loss = nn.MSELoss(reduction='none')
    # mse_loss = nn.SmoothL1Loss(reduction='none')

    for epoch in range(value_net_epoch):
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seq_len = seqs.size(1)
            seq_dim = seqs.size(2)
            
            seqs = seqs.to(device) # (batch_size, seq_len, seq_dim)
            labels = labels.to(device)
            # weights = weights.unsqueeze(-1).to(device)
            weights = weights.to(device)
            
            fake_seqs = generator.sample(batch_size, labels, lengths) 
            
            fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
            d_scores = discriminator.forward(labels, fake_seqs_wv, lengths).detach()
            
            v_pred = value_net(labels, fake_seqs_wv, lengths) # (batch_size, seq_len, 1)
            if v_pred.dim() == 3:
                v_pred = v_pred.squeeze(-1)  # (batch_size, seq_len)
                d_scores = d_scores.squeeze(-1)
                
            # d_target = d_scores.view(batch_size, 1).expand(-1, v_pred.size(1))  # (batch_size, seq_len)
            # fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
            # real_validity = discriminator.forward(labels, real_seqs_wv, lengths) * weights
            
            val_optimizer.zero_grad()
            
            max_T = v_pred.size(1) # seq_len
            lengths = lengths.to(device)
            mask = (torch.arange(max_T, device=device)[None, :] < lengths[:, None]).float()  # (batch_size, seq_len)

            # loss_v_all = mse_loss(v_pred, d_target) * mask * weights # (batch_size, seq_len)
            # v_loss = torch.sum(loss_v_all) / (torch.sum(mask) + 1e-8) # (batch_size, 1)
            # loss_all = mse_loss(v_pred, d_target)   # (B, T)
            # seq_loss_sum = (loss_all * mask).sum(dim=1)        # (B,)
            # seq_loss = seq_loss_sum / (mask.sum(dim=1) + 1e-8)        # (B,)                          # (B,)
            # v_target = (v_pred * mask).sum(dim = 1)
            # d_target = d_scores * lengths
            # seq_loss = mse_loss(v_target, d_target)
            # seq_loss = mse_loss(v_target , d_scores)
            v_target = v_pred * mask
            d_target = d_scores * mask
            seq_loss = ((v_target - d_target) ** 2).sum(dim = 1)/lengths
            v_loss = (seq_loss * weights).mean()
            
            # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            
            # with torch.backends.cudnn.flags(enabled=False):
            #     gp = compute_gradient_penalty(discriminator, real_seqs_wv, fake_seqs_wv,labels, lengths,device)
            # total_d_loss = d_loss + gp
            # total_d_loss.backward()
            v_loss.backward()
            val_optimizer.step()
            
        print('Pre-train Epoch [%d] Value Net Loss: %f'% (epoch, v_loss))
        
        torch.save(value_net.state_dict(), f'{model_path}valuenet_pre.pth')

def step_train(generator, discriminator,value_net, dataloader, epochs, device, seq_dim, n_d_critic, n_v_critic, model_path, checkpoint, real_data, bins_data, meta_attrs, sery_attrs, label_dict, n_roll):
    optimizer_g = optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=0.00005)
    optimizer_v = optim.Adam(value_net.parameters(), lr=0.0001)

    # rollout = Rollout(generator, 0.8)
    gan_loss = GANLoss(generator.x_list)
    # mse_loss = nn.MSELoss(reduction='none')
    # mse_loss = nn.SmoothL1Loss(reduction='none')

    for epoch in range(epochs):
        start = time.perf_counter()
        torch.cuda.empty_cache()
        for i, (seqs, labels, lengths, weights) in enumerate(dataloader):
            # batch_size = seqs.size(0)
            # seqs = seqs.to(device)
            # labels = labels.to(device)
            # weights = weights.unsqueeze(-1).to(device)
            
            # samples = generator.sample(batch_size, labels, lengths) # (batch_size, seq_len, seq_dim)
            # zeros = torch.zeros((batch_size, 1, seq_dim)).type(torch.LongTensor).to(device)
            # inputs = torch.cat([zeros, samples], dim=1)[:,:-1,:].contiguous()
            # targets = samples.contiguous().view(-1,seq_dim)
            
            # rewards = rollout.get_reward(samples, n_roll, discriminator, labels, lengths)
            # rewards_exp = rewards.clone().contiguous().view((-1,)).to(device)
            
            # prob = generator.forward(labels, lengths, inputs, samples) # (batch_size, seq_len, prob_dim)
            
            # g_loss = gan_loss.forward(prob, targets, rewards_exp, device, weights)
            
            # optimizer_g.zero_grad()
            # g_loss.backward()
            # optimizer_g.step()
            # rollout.update_params()
            
            batch_size = seqs.size(0)
            seqs = seqs.to(device)
            labels = labels.to(device)
            weights = weights.unsqueeze(-1).to(device)
            
            samples = generator.sample(batch_size, labels, lengths) # (batch_size, seq_len, seq_dim)
            zeros = torch.zeros((batch_size, 1, seq_dim)).type(torch.LongTensor).to(device)
            inputs = torch.cat([zeros, samples], dim=1)[:,:-1,:].contiguous()
            targets = samples.contiguous().view(-1,seq_dim) # (batch_size * seq_len, seq_dim)
            
            # rewards = value_net.forward(labels, discriminator.seq2wv(samples).to(device), lengths) # (batch_size, seq_len, 1)
            # rewards = discriminator.forward(labels, discriminator.seq2wv(samples).to(device), lengths).detach() # (batch_size, seq_len, 1)
            # baseline = value_net.forward(labels, discriminator.seq2wv(samples).to(device), lengths).detach()
            
            # rewards = rewards - baseline
            
            # if rewards.dim() == 3:
            #     rewards = rewards.squeeze(-1) # (batch_size, seq_len)
            with torch.no_grad():
                d_scores = discriminator.forward(labels, discriminator.seq2wv(samples).to(device), lengths)  # (B, T, 1) or (B, T)
                v_pred   = value_net.forward(labels, discriminator.seq2wv(samples).to(device), lengths)      # (B, T, 1) or (B, T)

            # squeeze last dim if present
            if d_scores.dim() == 3:
                d_scores = d_scores.squeeze(-1)
            if v_pred.dim() == 3:
                v_pred = v_pred.squeeze(-1)
            rewards = d_scores - v_pred
            # print(rewards)
            
            prob = generator.forward(labels, lengths, inputs, samples) # (batch_size, seq_len, prob_dim)
            
            # print(prob)
            
            max_T = rewards.size(1)
            length_devs = lengths.to(device)
            mask = (torch.arange(max_T, device=device)[None, :] < length_devs[:, None]).float() # (batch_size, seq_len)
            
            rewards *= mask
            rewards_exp = rewards.clone().contiguous().view((-1,)).to(device).detach() 
            
            g_loss = gan_loss.forward(prob, targets, rewards_exp, device, weights)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            for _ in range(n_d_critic):
                optimizer_d.zero_grad()
                fake_seqs = generator.sample(batch_size, labels, lengths)

                real_seqs_wv = discriminator.seq2wv(seqs).to(device)
                fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
                
                # fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
                # real_validity = discriminator.forward(labels, real_seqs_wv, lengths) * weights
                real_validity = discriminator.forward(labels, real_seqs_wv, lengths)
                fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths)
                
                
                
                
                # max_T = v_pred.size(1) # seq_len
                # length_devs = lengths.to(device)
                # mask = (torch.arange(max_T, device=device)[None, :] < length_devs[:, None]).float()  # (batch_size, seq_len)
                
                # fake_validity_target = fake_validity * mask
                # real_validity_target = real_validity * mask
                real_validity_mean = discriminator.cal_value(lengths,real_validity) * weights
                fake_validity_mean = discriminator.cal_value(lengths,fake_validity) * weights
                
            
                # d_loss_real = -torch.mean(real_validity)
                # d_loss_fake = torch.mean(fake_validity)
                d_loss_real = -torch.mean(real_validity_mean)
                d_loss_fake = torch.mean(fake_validity_mean)
                
                d_loss = d_loss_real + d_loss_fake

                
                # start_time = time.perf_counter()
                with torch.backends.cudnn.flags(enabled=False):
                    gp = compute_gradient_penalty(discriminator, real_seqs_wv, fake_seqs_wv,labels, lengths,device)
                # grad_time += time.perf_counter() - start_time
                
                # start_time = time.perf_counter()
                total_d_loss = d_loss + gp
                total_d_loss.backward()
                
                optimizer_d.step()
                # other_time += time.perf_counter() - start_time
                
            for _ in range(n_v_critic):
                optimizer_v.zero_grad()
                fake_seqs = generator.sample(batch_size, labels, lengths)
                fake_seqs_wv = discriminator.seq2wv(fake_seqs.detach()).to(device)
                # fake_validity = discriminator.forward(labels, fake_seqs_wv, lengths) * weights
                
                d_scores = discriminator.forward(labels, fake_seqs_wv, lengths).detach()
                # d_scores = d_scores.view(-1)
            
                v_pred = value_net(labels, fake_seqs_wv, lengths) # (batch_size, seq_len, 1)
                if v_pred.dim() == 3:
                    v_pred = v_pred.squeeze(-1)  # (batch_size, seq_len)
                    d_scores = d_scores.squeeze(-1)
                
                # d_target = d_scores.view(batch_size, 1).expand(-1, v_pred.size(1))  # (batch_size, seq_len)
                
                max_T = v_pred.size(1) # seq_len
                length_devs = lengths.to(device)
                mask = (torch.arange(max_T, device=device)[None, :] < length_devs[:, None]).float()  # (batch_size, seq_len)

                # loss_v_all = mse_loss(v_pred, d_target) * mask * weights # (batch_size, seq_len)
                # v_loss = torch.sum(loss_v_all) / (torch.sum(mask) + 1e-8) # (batch_size, 1)
                v_target = v_pred * mask
                d_target = d_scores * mask
                seq_loss = ((v_target - d_target) ** 2).sum(dim = 1)/length_devs
                # seq_loss = mse_loss(v_target, d_target)
                # loss_all = mse_loss(v_pred, d_target)   # (B, T)
                # seq_loss_sum = (loss_all * mask).sum(dim=1)        # (B,)
                # seq_loss = seq_loss_sum / (mask.sum(dim=1) + 1e-8)        # (B,)                          # (B,)
                v_loss = (seq_loss * weights).mean()
                
                v_loss.backward()
                optimizer_v.step()
                
            # print("sample_time: ", sample_time, "reward_time: ", reward_time, "g_forward_time: ", g_forward_time, "l_forward_time: ", l_forward_time, "d_forward_time: ", d_forward_time, "grad_time: ", grad_time, "other_time: ", other_time)
        end = time.perf_counter()
        
        
        fake_data = get_fake_json(label_dict, len(label_dict), real_data, generator, bins_data, sery_attrs, meta_attrs, batch_size)
        eu_sum = cal_euler(real_data, fake_data, label_dict)
        hm_sum = cal_hamming(real_data, fake_data, label_dict)

        # print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()} ({d_loss_real.item()}, {d_loss_fake.item()}), G Loss: {g_loss.item()}, V loss: {v_loss.item()}, time cost: {end - start}, Euler distance: {eu_sum}, Hamming distance: {hm_sum}.")
        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()} ({d_loss_real.item()}, {d_loss_fake.item()}), G Loss: {g_loss.item()}, time cost: {end - start}, Euler distance: {eu_sum}, Hamming distance: {hm_sum}.")

        torch.save(generator.state_dict(), f'{model_path}generator.pth')
        torch.save(discriminator.state_dict(), f'{model_path}discriminator.pth')
        # torch.save(value_net.state_dict(), f'{model_path}valuenet.pth')

        if (epoch + 1) % checkpoint == 0:
            torch.save(generator.state_dict(), f'{model_path}generator_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'{model_path}discriminator_{epoch+1}.pth')
            # torch.save(value_net.state_dict(), f'{model_path}valuenet_{epoch+1}.pth')
        
        torch.cuda.empty_cache()

# %%
def model_train(label_dict, dataset, json_folder, bins_folder, wordvec_folder, model_folder,
                meta_attrs, sery_attrs, 
                model_paras):
    label_dim = len(label_dict)
    batch_size = model_paras['batch_size']
    epochs = model_paras['epoch']
    max_seq_len = model_paras['max_seq_len']
    series_word_vec_size = model_paras['series_word_vec_size']
    meta_word_vec_size = model_paras['meta_word_vec_size']
    n_critic = model_paras['n_critic']
    n_roll = model_paras['n_roll']
    checkpoint = model_paras['checkpoint']
    pre_trained_generator_epoch = model_paras['pre_trained_generator_epoch']
    pre_trained_discriminator_epoch = model_paras['pre_trained_discriminator_epoch']
    
    seq_dim = len(sery_attrs) + len(meta_attrs)
    
    data_folder = f'./{json_folder}/{dataset}/'
    bins_file_name = f'./{bins_folder}/bins_{dataset}.json'
    wordvec_file_name = f'./{wordvec_folder}/word_vec_{dataset}.json'
    model_folder_name = f'./{model_folder}/{dataset}/'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(wordvec_file_name, 'r') as f:
        wv_dict = json.load(f)
    
    wv = {}
    for key, metrics in wv_dict.items():
        wv[key] = torch.tensor(metrics, dtype=torch.float32).to(device)

    x_list = [wv_tensor.size(0) for wv_tensor in wv.values()]
    print(x_list)
   

    generator = Generator(label_dim,seq_dim,max_seq_len,x_list,device)
    discriminator = Discriminator(label_dim, series_word_vec_size* len(sery_attrs) + meta_word_vec_size * len(meta_attrs) ,max_seq_len,x_list,wv,device)
    value_net = ValueNet(label_dim, series_word_vec_size* len(sery_attrs) + meta_word_vec_size * len(meta_attrs), max_seq_len,x_list,device)
    
    generator.to(device)
    discriminator.to(device)
    value_net.to(device)

    print(device)

    print("Process dataset...")

    dataset = CustomDataset(data_folder = data_folder, bins_file=bins_file_name, class_mapping=label_dict,max_seq_len=max_seq_len, meta_attrs=meta_attrs,sery_attrs=sery_attrs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # checkpoint_g = torch.load(f'{model_folder_name}generator_pre.pth')  
    # generator.load_state_dict(checkpoint_g) 

    # checkpoint_d = torch.load(f'{model_folder_name}discriminator_pre.pth') 
    # discriminator.load_state_dict(checkpoint_d) 
    
    # checkpoint_v = torch.load(f'{model_folder_name}valuenet_pre.pth') 
    # value_net.load_state_dict(checkpoint_v) 
    
    # print("Pre-training...")
    # pre_train(generator, discriminator, dataloader, pre_trained_generator_epoch, pre_trained_discriminator_epoch, device, model_folder_name)

    # print("Trainning...")
    # bins_data = dataset.bins_data
    # real_data = get_real_data(data_folder, label_dict, meta_attrs, sery_attrs, bins_data, max_seq_len)
    # train(generator, discriminator, dataloader, epochs, device, seq_dim, n_roll, n_critic, model_folder_name, checkpoint, real_data, bins_data, meta_attrs, sery_attrs, label_dict)
    
    print("Pre-training...")
    pre_trained_valuenet_epoch = 1
    step_pre_train(generator, discriminator, value_net, dataloader, pre_trained_generator_epoch, pre_trained_discriminator_epoch, pre_trained_valuenet_epoch, device, model_folder_name)
    # step_pre_train(generator, discriminator, dataloader, pre_trained_generator_epoch, pre_trained_discriminator_epoch, pre_trained_valuenet_epoch, device, model_folder_name)

    print("Trainning...")
    n_v_critic = 1
    bins_data = dataset.bins_data
    real_data = get_real_data(data_folder, label_dict, meta_attrs, sery_attrs, bins_data, max_seq_len)
    
    step_train(generator, discriminator, value_net, dataloader, epochs, device, seq_dim, n_critic, n_v_critic, model_folder_name, checkpoint, real_data, bins_data, meta_attrs, sery_attrs, label_dict, n_roll)
    # step_train(generator, discriminator, dataloader, epochs, device, seq_dim, n_critic, n_v_critic, model_folder_name, checkpoint, real_data, bins_data, meta_attrs, sery_attrs, label_dict, n_roll)