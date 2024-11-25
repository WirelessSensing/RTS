#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F



class ImagingModule(nn.Module):
    def __init__(self,feature_length,img_width,img_height) -> None:
        super(ImagingModule,self).__init__()
        self.feature_length = feature_length
        self.img_width = img_width
        self.img_height = img_height
        self.Fc1 = nn.Linear(feature_length,1600)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,
                               kernel_size=(3,3),stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(1,1),stride=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=1)
        self.maskAttention = nn.Parameter(torch.ones(size=[40,40]),requires_grad=True)

    def forward(self,X):
        res = self.relu(self.Fc1(X))
        res = res.reshape(-1,1,40,40)
        res = res * self.maskAttention
        res = F.interpolate(res,scale_factor=3,mode="bilinear")
        res = self.relu(self.conv1(res)) # 120,120
        res = F.interpolate(res,scale_factor=3,mode="bilinear")
        res = self.relu(self.conv2(res)) # 360,360
        res = self.conv3(res)
        return res #把X也返回来
    

class Encoder(nn.Module):
    def __init__(self,in_channels) -> None:
        super(Encoder,self).__init__()
        self.in_channels = in_channels
    
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=128,kernel_size=(3,3),stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(6,5),stride=1),
            
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=256,kernel_size=(9,9),stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(7,7),stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(2,1),stride=1),
            
        )
       
        
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=512,kernel_size=(16,15),stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(1,1),stride=1),
            
        )
    
    def forward(self,data):
        '''
        @param data:  BatchSzie, 16, 16, 16
        '''
        local_feature_map = self.local_conv(data)
        middle_feature_map = self.mid_conv(data)
        global_feature_map = self.global_conv(data)
        res_feature = local_feature_map + middle_feature_map + global_feature_map # 
        res_feature = torch.flatten(res_feature,start_dim=1)
        return res_feature
    
class RadioNet(nn.Module):
    def __init__(self,in_channels=16,feature_length=1024,img_width=360,img_height=360) -> None:
        super(RadioNet,self).__init__()
        self.in_channels = in_channels
        self.feature_length = feature_length
        self.encoder = Encoder(in_channels=in_channels)
        self.imgnet = ImagingModule(feature_length=feature_length,img_width=img_width,img_height=img_height)
    
    def forward(self,X):
        feature = self.encoder(X)
        imgres = self.imgnet(feature)
        return imgres.squeeze(1) 
    


