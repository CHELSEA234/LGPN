import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

from torch import linalg as LA
# from torchvision.models import resnet18, ResNet18_Weights
# from torchvision.models import resnet50, ResNet50_Weights
from copy import deepcopy
from resnet_stem import resnet50 as GenTraceCapture

class GCN_refinement(nn.Module):
    def __init__(self,
                layer_num=2,
                in_dim=64,
                feat_dim=64,
                output_dim=64,
                alpha=0.2,
                attn=True,
                self_attn=False,
                matching_matrix_list=None,
                matrix_scale=1000,
                ):
        super(GCN_refinement, self).__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.gcn_drop = nn.Dropout(0.2)
        self.W = nn.ModuleList()
        self.corr_list = nn.ModuleList()
        self.match_lst = nn.ModuleList()
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.alpha = alpha
        self.attn  = attn
        self.self_attn = self_attn

        for layer in range(layer_num*3):
            input_dim = self.in_dim if layer == 0 else self.feat_dim
            self.W.append(nn.Linear(input_dim, self.feat_dim))

        self.output_mlp = nn.Linear(feat_dim, output_dim)
        self.a = nn.Parameter(torch.empty(size=(2*output_dim, 1)))
        self.attn_fuc = self._prepare_attentional_mechanism_input
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.activ_out = nn.Tanh()

        ## graph pooling layer:
        ## three size values: 6, 18, 43
        self.SCALE = matrix_scale
        self.corr_list.append(nn.Linear(self.feat_dim, 18))
        self.corr_list.append(nn.Linear(self.feat_dim, 55))
        self.match_lst.append(nn.Linear(self.feat_dim, 18))
        self.match_lst.append(nn.Linear(self.feat_dim, 55))
        self.corre_activ = nn.Sigmoid()
        self.match_activ = nn.Softmax(dim=1)

    def _prepare_attentional_mechanism_input(self, Wh, dot=True):
        mul_a = self.a[:self.output_dim, :]
        mul_a = torch.unsqueeze(mul_a, 0)
        mul_a = torch.cat([mul_a]*Wh.size()[0], axis=0)
        Wh1 = Wh.bmm(mul_a)

        mul_b = self.a[self.output_dim:, :]
        mul_b = torch.unsqueeze(mul_b, 0)
        mul_b = torch.cat([mul_b]*Wh.size()[0], axis=0)
        Wh2 = Wh.bmm(mul_b)            
        Wh2 = torch.permute(Wh2, (0,2,1))
        e = Wh1 + Wh2
        return e, -9e15*torch.ones_like(e)

    def _GCN_loop(self, x, adj, idx):
        '''
            the standard GCN operation for the AXW with the self-loop.
            attention is optional.
        '''
        if self.attn:
            e, zero_vec = self._prepare_attentional_mechanism_input(x)
            adj_ab = torch.where(adj > 0, e, zero_vec)
            adj    = F.softmax(adj_ab, dim=2)
        
        denom = adj.sum(2).unsqueeze(2) + 1 
        for l in range(self.layer_num):
            Ax = adj.bmm(x)
            AxW = self.W[l+idx*2](Ax)
            B   = self.W[l+idx*2](x)
            AxW = AxW + B       # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            x = self.gcn_drop(gAxW) if l < self.layer_num - 1 else gAxW
        print(f"output from the gcn loop. ", adj.size())
        return x

    def _matching_gen(self, x, adj, idx):
        '''
        generate matching matrix, which is a soft assignment of each node at this layer to the next layer.
        generate the adjcency matrix in the next level.
        '''
        match_matrix = self.match_lst[idx](x)
        match_matrix = self.match_activ(match_matrix)
        match_matrix = torch.permute(match_matrix, (0,2,1))
        x_coarse = match_matrix.bmm(x)
        corr_graph = self.corr_list[idx](x)
        corr_graph = self.corre_activ(self.SCALE*corr_graph)
        tmp = adj.bmm(corr_graph)
        corr_graph = torch.permute(corr_graph, (0,2,1))
        adj_pred   = corr_graph.bmm(tmp)
        zero_vec = -9e15*torch.ones_like(adj_pred)
        ones_vec = torch.ones_like(adj_pred)
        adj      = torch.where(adj_pred > 0.1, ones_vec, zero_vec)

        return x_coarse, adj, match_matrix, corr_graph

    def forward(self, x, adj_0):
        x_0 = self._GCN_loop(x, adj_0, idx=0)
        
        x_1, adj_1, mm_1, cg_1 = self._matching_gen(x_0, adj_0, idx=0)
        x_1 = self._GCN_loop(x_1, adj_1, idx=1)

        x_2, adj_2, mm_2, cg_2 = self._matching_gen(x_1, adj_1, idx=1)
        x_2 = self._GCN_loop(x_2, adj_2, idx=2)

        output = self.output_mlp(x_2+x_0)
        output = self.activ_out(output)
        output_intermediate = x_1
        return output, output_intermediate, [mm_1, mm_2, cg_1, cg_2], [adj_0, adj_1, adj_2]

class CatDepth(nn.Module):
    def __init__(self):
        super(CatDepth, self).__init__()

    def forward(self, x, y):
        return torch.cat([x,y],dim=1)

class LGP_Net(nn.Module):
    def __init__(self, 
                class_num, 
                task_num, 
                feat_dim=512, 
                output_dim=128,
                gcn_layer=2,
                gcn_feat_dim=128,
                matching_matrix_list=None
                ):
        super(LGP_Net, self).__init__()
        self.GTC = GenTraceCapture(pretrained=True)
        self.GTC.fc = nn.Linear(2048, 512)
        self.concat_depth = CatDepth()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_1x1_merge = nn.Sequential(nn.Conv2d(128, 64,
                                                  kernel_size=1, stride=1,
                                                  bias=False,groups=2),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.2)
                                       )
        self.feat_dim   = feat_dim
        self.output_dim = output_dim
        self.task_num   = task_num
        self.GCN_refine = GCN_refinement(
                                        layer_num=gcn_layer,
                                        in_dim=gcn_feat_dim,
                                        feat_dim=gcn_feat_dim,
                                        output_dim=gcn_feat_dim,
                                        matching_matrix_list=matching_matrix_list
                                        )
        for i in range(task_num):
            name = f'head_{(i+1):d}'
            setattr(self, 
                    name, 
                    self._make_pred_head()
                    )
        self.classifier = self._make_classifier(input_dim=output_dim)
        self.classifier_real_fake = self._make_classifier(input_dim=(512+128))

    def _make_pred_head(self):
        return nn.Sequential(
                            nn.Linear(self.feat_dim, self.feat_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.feat_dim, self.output_dim)
                            )

    def _make_classifier(self, input_dim, output_num=2):
        return nn.Sequential(
                            nn.Linear(input_dim, input_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(input_dim, output_num),
                            )

    def forward(self, x, adj, tsne=False):
        output_feat = self.GTC(x)
        res_list = []
        for i in range(self.task_num):	## self.task_num is 55.
            name = f'head_{(i+1):d}'
            x = getattr(self, name)(output_feat)
            x = F.normalize(x, dim=1)
            res_list.append(torch.unsqueeze(x, 1))

        gcn_inputs = torch.cat(res_list, dim=1)	## torch.Size([64, 55, 128])
        gcn_out, gcn_middle, [mm_1, mm_2, cg_1, cg_2], [adj_0, adj_1, adj_2] = self.GCN_refine(gcn_inputs, adj)
        gcn_middle = torch.mean(gcn_middle, axis=1)

        feature_real_fake = torch.cat([gcn_middle, output_feat], axis=-1)
        pred_real_fake = self.classifier_real_fake(feature_real_fake)
        ce_pred = self.classifier(gcn_out)
        if not tsne:
            return pred_real_fake, ce_pred, gcn_out, [mm_1, mm_2, cg_1, cg_2]
        else:
            return pred_real_fake, ce_pred, [adj_0, adj_1, adj_2], [mm_1, mm_2, cg_1, cg_2]