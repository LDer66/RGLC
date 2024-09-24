from torch_geometric.nn import MLP
import torch.nn as nn
import torch

class ConNN(nn.Module):
    def __init__(self, in_feats, out_feats, enc1:nn.Module, enc2:nn.Module, eta=1.0, lam=1.0) -> None:
        super().__init__()
        self.eta = eta
        self.in_feats = in_feats
        self.enc1 = enc1
        if lam is not None and lam > 0:
            self.lam = lam
            self.enc2 = enc2
            # self.proj = MLP([in_feats, out_feats], dropout=0.5)
            self.proj = nn.Sequential(nn.Linear(in_feats, in_feats), nn.ReLU(inplace=True), nn.Linear(out_feats, out_feats))
            self.init_emb()
        else:
            self.lam = None

    def forward(self, batchgraph):
        out1 = self.enc1(batchgraph)
        self.hidden_rep = self.enc1.hidden_rep
        self.hidden_rep1 = self.enc1.hidden_rep1
        out2=None
        if self.lam is not None:
            enc2 = self.get_ran_enc(self.enc1, self.enc2)
            out2 = enc2(batchgraph)
            self.z1 = self.proj(self.hidden_rep1)
            self.z2 = self.proj(enc2.hidden_rep1)
        return out1,out2
    

    def init_emb(self):
        for m in self.proj.modules():  # 只对 self.proj 中的线性层进行初始化
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)



    def get_ran_enc(self, enc1, enc2):
        for (adv_name,adv_param), (name,param) in zip(enc2.named_parameters(), enc1.named_parameters()):

            adv_param.data = param.data + self.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(enc1.device)
        return enc2

    def con_loss(self):
        if self.lam is None:
            return 0
        x, x_aug = self.z2, self.z1
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return self.lam * loss
    

    def con_loss1(self, labels):
        if self.lam is None:
            return 0
        x, x_aug = self.z2, self.z1 
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(x.device)
        mask.fill_diagonal_(0)

        if mask.sum() == 0:
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = -torch.log(loss).mean()
        else:
            pos_sim = (sim_matrix * mask).sum(dim=1)
            sim_sum = sim_matrix.sum(dim=1, keepdim=True) - sim_matrix.diagonal().unsqueeze(1)
            loss = (pos_sim / (sim_sum + 1e-8)).clamp(min=1e-8)
            loss = -torch.log(loss).mean()
            
        return self.lam * loss


    def get_used_parameters(self):
        if self.lam is not None:
            return list(self.enc1.parameters()) + list(self.proj.parameters())
        else:
            return self.parameters()
