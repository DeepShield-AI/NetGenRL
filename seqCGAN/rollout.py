import copy
import torch
import time
from seqCGAN.discriminator import Discriminator

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.device = model.device

    def get_reward(self, x, num, discriminator, label, length): # num is the sample times of rollouts
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        sample_time = 0
        seq2wv_time = 0
        forward_time = 0
        
        # num = 16
        for l in range(1, seq_len):
            labels = label.repeat(num, 1).to(self.device)
            lengths = length.repeat(num)
            xs = x.repeat(num, 1, 1).to(self.device)
            
            start_time = time.perf_counter()
            samples = self.own_model.sample(batch_size * num, labels, lengths, xs[:, :l, :])
            sample_time += time.perf_counter() - start_time
                
            start_time = time.perf_counter()
            samples_wv = discriminator.seq2wv(samples)
            seq2wv_time += time.perf_counter() - start_time
                
            start_time = time.perf_counter()
            pred = discriminator.forward(labels, samples_wv, lengths)
            forward_time += time.perf_counter() - start_time

            pred_slice = pred.clone().view(-1)
            
            for i in range(num):
                if i == 0:
                    rewards.append(pred_slice[i*batch_size:(i+1)*batch_size])
                else:
                    rewards[l-1] += pred_slice[i*batch_size:(i+1)*batch_size]
            samples_wv = discriminator.seq2wv(xs)
            pred = discriminator.forward(labels, samples_wv, lengths)
            pred_slice = pred.clone().view(-1)
            
        for i in range(num):
            if i == 0:
                rewards.append(pred_slice[i*batch_size:(i+1)*batch_size])
            else:
                rewards[seq_len-1] += pred_slice[i*batch_size:(i+1)*batch_size]
                
        rewards = torch.stack(rewards, dim=1) / (1.0 * num) 
        mask = torch.zeros(rewards.shape).to(self.device)
        for l in range(seq_len):
            mask[:, l] = length.gt(l).float()
        rewards *= mask
        return rewards
    
    # def get_reward(self, x, num, discriminator, label, length): # num is the sample times of rollouts
    #     """
    #     Args:
    #         x : (batch_size, seq_len) input data
    #         num : roll-out number
    #         discriminator : discrimanator model
    #     """
    #     rewards = []
    #     batch_size = x.size(0)
    #     seq_len = x.size(1)
    #     sample_time = 0
    #     seq2wv_time = 0
    #     forward_time = 0
        
    #     # num = 16
    #     for l in range(1, seq_len):
    #         labels = label.repeat(num, 1).to(self.device)
    #         lengths = length.repeat(num)
    #         xs = x.repeat(num, 1, 1).to(self.device)
            
    #         start_time = time.perf_counter()
    #         samples = self.own_model.sample(batch_size * num, labels, lengths, xs[:, :l, :])
    #         sample_time += time.perf_counter() - start_time
                
    #         start_time = time.perf_counter()
    #         samples_wv = discriminator.seq2wv(samples)
    #         seq2wv_time += time.perf_counter() - start_time
                
    #         start_time = time.perf_counter()
    #         pred_step = discriminator.forward(labels, samples_wv, lengths)
    #         pred = discriminator.cal_value(lengths, pred_step)
    #         forward_time += time.perf_counter() - start_time

    #         pred_slice = pred.clone().view(-1)
            
    #         for i in range(num):
    #             if i == 0:
    #                 rewards.append(pred_slice[i*batch_size:(i+1)*batch_size])
    #             else:
    #                 rewards[l-1] += pred_slice[i*batch_size:(i+1)*batch_size]
    #         samples_wv = discriminator.seq2wv(xs)
    #         pred_step = discriminator.forward(labels, samples_wv, lengths)
    #         pred = discriminator.cal_value(lengths,pred_step)
    #         pred_slice = pred.clone().view(-1)
            
    #     for i in range(num):
    #         if i == 0:
    #             rewards.append(pred_slice[i*batch_size:(i+1)*batch_size])
    #         else:
    #             rewards[seq_len-1] += pred_slice[i*batch_size:(i+1)*batch_size]
                
    #     rewards = torch.stack(rewards, dim=1) / (1.0 * num) 
    #     mask = torch.zeros(rewards.shape).to(self.device)
    #     for l in range(seq_len):
    #         mask[:, l] = length.gt(l).float()
    #     rewards *= mask
    #     return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]