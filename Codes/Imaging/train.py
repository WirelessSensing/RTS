#encoding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloaderf import RTIDataSet
from backbone import RadioNet
import torch.optim as optim 
from torch.utils.data import DataLoader
torch.manual_seed(3447)


epochs = 100
lr = 5e-4
ImglossFunction = nn.MSELoss()

device = "cuda:0"

def saveValid(vaildfile,pretrainmodelfile,savedir):
    validDataset = RTIDataSet(vaildfile)
    validloader = DataLoader(validDataset,batch_size=256,num_workers=4,pin_memory=True,shuffle=False)
    pretrainmodel = torch.load(pretrainmodelfile)
    count = 0
    with torch.no_grad():
        for valididx,(validdata,_,_,_) in enumerate(validloader):
            print(valididx)
            validdata = validdata.to(device)
            valid_img = pretrainmodel(validdata) #BatchSize,120,120
            batchSize = valid_img.shape[0] 
            for ii in range(batchSize):
                #单个数据
                valid_img_single = valid_img[ii].to('cpu') #转换到CPU上面
                torch.save(valid_img_single,savedir+str(count))
                count += 1


def trainFF(trainfile):
    trainDataset = RTIDataSet(trainfile)
    trainloader = DataLoader(dataset=trainDataset,batch_size=128,shuffle=False,num_workers=4,pin_memory=True)
    RadioNetModel = RadioNet(in_channels=16,feature_length=1024,img_width=360,img_height=360).float().to(device)
    optimizer = optim.Adam(RadioNetModel.parameters(),lr=lr)
    for epoch in range(epochs):
        for idx,(traindata,trainground,_,_) in enumerate(trainloader): 
            traindata = traindata.to(device)
            trainground = trainground.to(device)
            train_img = RadioNetModel(traindata)
            train_img_loss = ImglossFunction(train_img,trainground)
            allLoss = train_img_loss 
            optimizer.zero_grad()
            allLoss.backward()
            optimizer.step()

        if epoch == epochs - 1:
            torch.save(RadioNetModel,"../Records/RadioNet_Img.pth")

            

    

  