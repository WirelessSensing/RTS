#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from S2_arch import DiffIRS2
import torch.optim as optim
from dataloader import RTIDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from EvaluatePixelIoU import PixelCount, IoUCount
import numpy as np
from associate import processData
from evaluateN import LoadFileForEva
import sys
import math


device = "cpu"
lr = 1e-4
epochs = 12

def visualize(testfile,sencondPretrainfile):
    vifpvalue = 0
    vifcount = 0
    SecondStage_pretrain = torch.load(sencondPretrainfile)
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=2,shuffle=False,num_workers=4,pin_memory=True)
    with torch.no_grad():
        SecondStage_pretrain.eval()
        for testidx,(test_data,test_ground) in enumerate(testloader):
            test_data = test_data.to(device)
            test_ground = test_ground.to(device)
            test_img = test_data
            test_img = test_img.unsqueeze(1)
            test_res = SecondStage_pretrain(test_img,None)
            test_res = test_res.squeeze(1)
            if testidx < 120:
                for kk in range(test_res.shape[0]):
                    imgip = test_res[kk].detach().cpu().numpy()
                    imgt = test_ground[kk].detach().cpu().numpy()
                    plt.imsave("../visualize/"+str(testidx)+"_"+str(kk)+".png",imgip)
                    plt.imsave("../visualize/"+str(testidx)+"_"+str(kk)+"_gt.png",imgt)


def pxielCount(testfile,pretrainmodel):
    model = torch.load(pretrainmodel,map_location=device)
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=4,shuffle=False,num_workers=1,pin_memory=True)
    count = 0
    pixel_difference_sum = 0.0
    pixel_ratio_difference_sum = 0.0
    Iou_sum = 0.0
    str_count = 0
    with torch.no_grad():
        for testidx,(testdata,testground) in enumerate(testloader):
            print("test idx: ",testidx)
            testdata = testdata.to(device)
            testdata = testdata.unsqueeze(1)
            testground = testground.to(device)
            test_res = model(testdata,None)
            test_res = test_res.squeeze(1)
            batchSize = testdata.shape[0]
            test_res_cur = []
            for ii in range(test_res.shape[0]):
                test_res_cur.append(torch.from_numpy(processData(test_res[ii].detach().cpu().numpy())))
            test_res = torch.stack(test_res_cur,dim=0).to(device) #BatchSize,360,360           
            count += batchSize
            pixel_difference,pixel_ratio_difference = PixelCount(test_res,testground)
            pixel_difference_sum += pixel_difference
            pixel_ratio_difference_sum += pixel_ratio_difference
            Iou_sum += IoUCount(test_res,testground)
        
        pixel_difference_mean = pixel_difference_sum / count
        pixel_ratio_difference_mean = pixel_ratio_difference_sum / count
        Iou_mean = Iou_sum / count
        EDE_error = math.sqrt(pixel_difference_mean * 4 / math.pi) * 2 / 10

        with open("EvaluationRes.txt","a+",encoding="utf-8") as fe:
            fe.write("EDE:")
            fe.write("\t")
            fe.write(str(EDE_error))
            fe.write("\t")
            fe.write("RPD:")
            fe.write("\t")
            fe.write(str(pixel_ratio_difference_mean))
            fe.write("\t")
            fe.write("IoU:")
            fe.write("\t")
            fe.write(str(Iou_mean))
            fe.write("\n")
        
def evaluate(testfile,pretrainmodel):
    testDataset = RTIDataset(testfile)
    testloader = DataLoader(dataset=testDataset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)
    SecondStage = torch.load(pretrainmodel,map_location=device)
    ssim_sum_u = 0.0

    count = 0
    with torch.no_grad():
        SecondStage.eval()
        for testidx,(testdata,testlabel) in enumerate(testloader):
            print(testidx)
            testdata = testdata.to(device)
            testlabel = testlabel.to(device)
            testimg = testdata.unsqueeze(1)
            test_out = SecondStage(testimg,None)
            test_out = test_out.squeeze(1)
            batchSize = testdata.shape[0]
            for kk in range(batchSize):
                test_res = test_out[kk].to('cpu').numpy()
                test_test = testlabel[kk].to('cpu').numpy()
                ssim_batcht = LoadFileForEva(test_test,test_res)
                ssim_sum_u += ssim_batcht
                count += 1
        ssim_avg_u = ssim_sum_u / count
        print(ssim_avg_u)
        with open("EvaluationRes.txt","a+",encoding="utf-8") as fe:
            fe.write("SSIM:")
            fe.write("\t")
            fe.write(str(ssim_avg_u))
            fe.write("\n")




if __name__ == "__main__":
    pxielCount("../../datafiles/Test.txt","../PretrainModel/PreTrain.pth")
    evaluate("../../datafiles/Test.txt","../PretrainModel/PreTrain.pth")
    