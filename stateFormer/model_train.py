import torch
import torch.optim as optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader
from torch import nn
from stateFormer.transformer import TransGenerator
from stateFormer.dataset import CustomDataset
import time

def transformer_train(generator, dataloader, epochs, device, model_path, checkpoint):
    # train generator
    gen_optimizer = optim.Adam(generator.parameters(),lr=0.0001, betas=(0.5, 0.999))
    x_list = generator.x_list
    
    for epoch in range(epochs):
        start = time.perf_counter()
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
            
        end = time.perf_counter()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {g_loss.item()}, time cost: {end - start}.")

        torch.save(generator.state_dict(), f'{model_path}generator.pth')
        
        if (epoch + 1) % checkpoint == 0:
            torch.save(generator.state_dict(), f'{model_path}generator_{epoch+1}.pth')

# %%
def model_train(label_dict, dataset, json_folder, bins_folder, model_folder,
                meta_attrs, sery_attrs, 
                model_paras):
    label_dim = len(label_dict)
    batch_size = model_paras['batch_size']
    epochs = model_paras['epoch']
    max_seq_len = model_paras['max_seq_len']
    checkpoint = model_paras['checkpoint']
    
    seq_dim = len(sery_attrs) + len(meta_attrs)
    
    data_folder = f'./{json_folder}/{dataset}/'
    bins_file_name = f'./{bins_folder}/bins_{dataset}.json'
    model_folder_name = f'./{model_folder}/{dataset}/'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(bins_file_name, 'r') as f_bin:
        bins_datas = json.load(f_bin)
        x_list = [len(bins_data["intervals"]) for bins_data in bins_datas.values()]
    print(x_list)
   

    generator = TransGenerator(label_dim,seq_dim,max_seq_len,x_list,device)
    
    generator.to(device)

    print(device)

    print("Process dataset...")

    dataset = CustomDataset(data_folder = data_folder, bins_file=bins_file_name, class_mapping=label_dict,max_seq_len=max_seq_len, meta_attrs=meta_attrs,sery_attrs=sery_attrs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Trainning...")
    transformer_train(generator,dataloader,epochs,device,model_folder_name,checkpoint)