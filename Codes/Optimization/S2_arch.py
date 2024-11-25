#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from S1_arch import DIRformer
from vit_pytorch import ViT
import sys
sys.path.append("../DDPM/")
from ddpm import DDPMModule as DDPM 

class CPIE2(nn.Module):
    def __init__(self,n_feats=256,resolution=360,patch_size=18):
        super(CPIE2,self).__init__()
        self.vitmodel = ViT(image_size=resolution,patch_size=patch_size,num_classes=n_feats,dim=256,depth=4,heads=8,mlp_dim=512,channels=1,dim_head=64)

    def forward(self,lq):
        X = lq
        img_condition = self.vitmodel(X)
        return img_condition
    

class ResMLP(nn.Module):  
    def __init__(self,n_feats = 512): 
        
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res
    
# 去噪模块 他就去噪了5步
class denoise(nn.Module):
    
    def __init__(self,n_feats=64, n_denoise_res = 5,timesteps=5):
        
        super(denoise, self).__init__()
        self.max_period=timesteps*10 #最大迭代数量500
        n_featsx4=4*n_feats #特征数量*4  
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1) 
        fea = self.resmlp(c)
        return fea 
    
class DiffIRS2(nn.Module):
    def __init__(self,inp_channels=1, 
        out_channels=1, 
        dim = 32,
        num_blocks = [1,1,1,1], 
        num_refinement_blocks = 1,
        heads = [2,2,2,2],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   
        n_denoise_res = 5, 
        linear_start= 0.1, 
        linear_end= 0.99,  
        timesteps = 10,sample_list=[2,2,2],resolution=360,path_size=18):
        super(DiffIRS2, self).__init__()

        self.G = DIRformer(        
        inp_channels=inp_channels, 
        out_channels=out_channels, 
        dim = dim,
        num_blocks = num_blocks, 
        num_refinement_blocks = num_refinement_blocks,
        heads = heads,
        ffn_expansion_factor = ffn_expansion_factor,
        bias = bias,
        LayerNorm_type = LayerNorm_type,   
        sample_list=sample_list
        )

        self.condition = CPIE2(n_feats=256,resolution=resolution,patch_size=path_size)
        self.denoise= denoise(n_feats=64, n_denoise_res=n_denoise_res,timesteps=timesteps)

        self.diffusion = DDPM(denoise=self.denoise, condition=self.condition ,n_feats=64,linear_start= linear_start,
            linear_end= linear_end, timesteps = timesteps)
        
    def forward(self, img, IPRS1=None):
        if self.training:
            
            IPRS2, pred_IPR_list=self.diffusion(img,IPRS1)
            sr = self.G(img, IPRS2)
            return sr,pred_IPR_list 
        
        else:
            IPRS2=self.diffusion(img) 
            sr = self.G(img, IPRS2)
            return sr
        




