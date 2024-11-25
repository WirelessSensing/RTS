#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
import numbers
import math
from vit_pytorch import ViT

def to_3d(x):
    return rearrange(x,'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
       
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

      
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

       
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
    def forward(self, x,k_v):
        
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1) 
        x = x*k_v1+k_v2  
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1) 
        x = F.gelu(x1) * x2 
        x = self.project_out(x) 
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        '''
        dim: 输入通道数量
        num_heads: Attention中heads的数量
        bias: 偏置项
        '''
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) 
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False), 
        )
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias) 
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias) 
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) 

    def forward(self, x,k_v):
        '''
        x: 
        k_v: 
        '''
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        x = x*k_v1+k_v2  

        qkv = self.qkv_dwconv(self.qkv(x)) 
        q,k,v = qkv.chunk(3, dim=1) 
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 

        q = torch.nn.functional.normalize(q, dim=-1) 
        k = torch.nn.functional.normalize(k, dim=-1) 

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1) 

        out = (attn @ v) 
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) 

        out = self.project_out(out) 
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
      
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type) 
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type) 
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias) 

    def forward(self, y):
        
        x = y[0]  
        k_v=y[1] 
        x = x + self.attn(self.norm1(x),k_v) 
        x = x + self.ffn(self.norm2(x),k_v) 
        return [x,k_v] 
    
class OverlapPatchEmbed(nn.Module):
   
    def __init__(self,in_c=1,embed_dim=48,bias=False):
        
        super(OverlapPatchEmbed,self).__init__()
        self.proj = nn.Conv2d(in_c,embed_dim,kernel_size=3,stride=1,padding=1,bias=bias)

    def forward(self,x):
        x = self.proj(x)
        return x
    


class DownSampleMaxPool(nn.Module):
    
    def __init__(self,n_feat,kernel_size) -> None:
        super(DownSampleMaxPool,self).__init__()
        self.body = nn.Sequential(nn.MaxPool2d(kernel_size),nn.Conv2d(n_feat,2*n_feat,kernel_size=3,stride=1,padding=1,bias=False))
    def forward(self,x):
        return self.body(x)


class UpsampleMaxPool(nn.Module):
    def __init__(self,n_feat,kernel_size) -> None:
        super(UpsampleMaxPool,self).__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=kernel_size,mode='bilinear',align_corners=True),nn.Conv2d(n_feat,n_feat//2,kernel_size=3,stride=1,padding=1,bias=False))

    def forward(self,x):
        return self.body(x)


class DIRformer(nn.Module):
    def __init__(self,inp_channels=1,out_channels=1,dim=48,num_blocks=[2,2,2,2],num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,
                 bias = False,LayerNorm_type = 'WithBias',sample_list=[2,2,2]):
        
        super(DIRformer,self).__init__()

        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = DownSampleMaxPool(dim,sample_list[0]) 
        
        
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = DownSampleMaxPool(int(dim*2**1),sample_list[1]) 
        
       
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = DownSampleMaxPool(int(dim*2**2),sample_list[2]) 
        

       
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = UpsampleMaxPool(int(dim*2**3),sample_list[-1])  
    
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        
        self.up3_2 = UpsampleMaxPool(int(dim*2**2),sample_list[-2]) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = UpsampleMaxPool(int(dim*2**1),sample_list[-3])  
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self,inp_img,k_v):
       
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1,_ = self.encoder_level1([inp_enc_level1,k_v])
        
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2,_ = self.encoder_level2([inp_enc_level2,k_v])
       
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3,_ = self.encoder_level3([inp_enc_level3,k_v])
        

        inp_enc_level4 = self.down3_4(out_enc_level3) 
               
        latent,_ = self.latent([inp_enc_level4,k_v]) 
        

        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        
        out_dec_level3,_ = self.decoder_level3([inp_dec_level3,k_v]) 
       
        inp_dec_level2 = self.up3_2(out_dec_level3)
        
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        
        out_dec_level2,_ = self.decoder_level2([inp_dec_level2,k_v]) 
    
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,_ = self.decoder_level1([inp_dec_level1,k_v])
        out_dec_level1,_ = self.refinement([out_dec_level1,k_v])
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1
    

class CPIE(nn.Module):
    def __init__(self,n_feates=256,resolution=360,path_size=18):
      
        super(CPIE,self).__init__()
        self.vitmodel = ViT(image_size=resolution,patch_size=path_size,num_classes=n_feates,dim=256,depth=4,heads=8,mlp_dim=512,channels=2,dim_head=64)

    def forward(self,gt,lq):
    
        X = torch.cat([gt,lq],dim=1)
       
        img_condition = self.vitmodel(X)
        condition = img_condition
        return condition

class DiffS1(nn.Module):
    def __init__(self,inp_channels=1,out_channels=1,dim=32,num_blocks=[1,1,1,1],num_refinement_blocks = 1,heads = [2,2,2,2],
                 ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias',
                 sample_list=[2,2,2],resolution=360,path_size=18):
        super(DiffS1,self).__init__()
        self.G = DIRformer(inp_channels=inp_channels,out_channels=out_channels,dim=dim,num_blocks=num_blocks,num_refinement_blocks=num_refinement_blocks,
                           heads=heads,ffn_expansion_factor=ffn_expansion_factor,bias=bias,LayerNorm_type=LayerNorm_type,sample_list=sample_list)
        self.E = CPIE(n_feates=256,resolution=resolution,path_size=path_size)  
    
    def forward(self,x,gt):
        if self.training:
            IPRS1 = self.E(x,gt)
            sr = self.G(x,IPRS1)
            return sr
        else:
            IPRS1 = self.E(x,gt)
            sr = self.G(x,IPRS1)
            return sr
        
    


