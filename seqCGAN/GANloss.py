import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self, x_list):
        super(GANLoss, self).__init__()
        self.x_list = x_list

    def forward(self, prob, target, reward, device, weights):
        """
        Args:
            prob: (batch_size, seq_len, prob_dim), torch Variable
            target : (batch_size, seq_len, seq_dim), torch Variable
            reward : (batch_size * seq_len), torch Variable
        """
        N = target.size(0)
        C = prob.size(2)
        
        count = 0
        prob_scatter = prob.clone().view(-1, C)
        for i,x_len in enumerate(self.x_list):
            one_hot = torch.zeros((N, x_len)).to(device)
            one_hot.scatter_(1, target.data[:,i].view((-1,1)), 1)
            one_hot = one_hot.type(torch.BoolTensor).to(device)
            if i == 0:
                loss = torch.masked_select(prob_scatter[:,count:x_len + count], one_hot) # (batch_size, seq_len)
            else:
                loss += torch.masked_select(prob_scatter[:,count:x_len + count], one_hot)
            count += x_len
            
        loss /= len(self.x_list)
        
        loss = loss * reward 
        loss = loss * weights
        loss = -torch.mean(loss)
        return loss