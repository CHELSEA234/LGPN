import os
import time
import torch
import torch.nn as nn

from torchvision.utils import make_grid
from tqdm import tqdm

device = torch.device('cuda:0')
device_ids = [0]

class IsolatingLossFunction(torch.nn.Module):
    def __init__(self, c, R, p=2):
        super().__init__()
        self.c = c.clone().detach() # Center of the hypershpere, c ∈ ℝ^d (d-dimensional real-valued vector)
        self.R = R.clone().detach() # Radius of the hypersphere, R ∈ ℝ^1 (Real-valued)
        self.p = p                  # norm value (p-norm), p ∈ ℝ^1 (Default 2)
        self.margin_natu = (0.15)*self.R    
        self.margin_mani = (1.85)*self.R
        self.threshold   = 0.5*(self.margin_mani+self.margin_natu)

        print('\n')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(f'The Radius manipul is {self.margin_natu}.')
        print(f'The Radius expansn is {self.margin_mani}.')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('\n')
        self.pdist = torch.nn.PairwiseDistance(p=self.p) # Creating a Pairwise Distance object
        self.dis_curBatch = 0
        self.dis = 0

    def forward(self, model_output, label, threshold_new=None, update_flag=None):
        ## Generate the mask.
        bs, feat_dim, w, h = model_output.size()
        model_output = model_output.permute(0,2,3,1)
        model_output = torch.reshape(model_output, (-1, feat_dim))
        dist = self.pdist(model_output, self.c)
        self.dist = dist
        pred_mask = torch.gt(self.dist, self.threshold).to(torch.float32)
        pred_mask = torch.reshape(pred_mask, (bs,w,h,1)).permute(0,3,1,2)
        self.dist = torch.reshape(self.dist, (bs,w,h,1)).permute(0,3,1,2)
        self.dis_curBatch = pred_mask.to(device).to(torch.float32)

        label = torch.reshape(label, (bs*w*h,1))
        label_sum = label.sum().item()

        label_nat  = torch.eq(label,0)
        label_mani = torch.eq(label,1)
        assert dist.size() == label_nat[:,0].size() 
        assert dist.size() == label_mani[:,0].size()

        label_nat_sum  = label_nat.sum().item()
        label_mani_sum = label_mani.sum().item()

        dist_nat  = torch.masked_select(dist, label_nat[:,0])
        dist_mani = torch.max(torch.tensor(0).to(device).float(),
                              torch.sub(self.margin_mani, 
                                        torch.masked_select(dist, label_mani[:,0]))
                              )

        loss_nat  = dist_nat.sum()/label_nat_sum if label_nat_sum != 0 else \
                    torch.tensor(0).to(device).float()
        loss_mani = dist_mani.sum()/label_mani_sum if label_mani_sum != 0 else \
                    torch.tensor(0).to(device).float()
        loss_total = loss_nat + loss_mani

        return loss_total.to(device), loss_mani.to(device), loss_nat.to(device)


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # LOSS_MAP = IsolatingLossFunction(center,radius).to(device)
    pdist = torch.nn.PairwiseDistance(p=2)
    feature_0 = torch.rand([100, 512])
    feature_1 = torch.rand([100, 512])
    dist = pdist(feature_0, feature_1)
    # print(type(dist))
    # print(dist.size())