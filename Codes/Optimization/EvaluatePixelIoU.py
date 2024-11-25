#encoding=utf-8

'''
用于评估像素数量和IoU比例
'''

import numpy as np
import torch


def CalGradDiffernece(prediction,groundTruth):

    prediction = prediction.cpu().numpy()
    groundTruth = groundTruth.cpu().numpy()
    value = 0.0
    data = torch.from_numpy(prediction)
    data = torch.where(data > value,1,0)
    data = data.numpy()


    difference = np.abs(np.sum(data) - np.sum(groundTruth))
    difference_ratio = difference / np.sum(groundTruth)

    return difference, difference_ratio

def PixelCount(prediction,groundTruth):
    difference_cur_batch = 0.0
    difference_ratio_cur_batch = 0.0
    for i in range(prediction.shape[0]):
        difference,difference_ratio = CalGradDiffernece(prediction[i],groundTruth[i])
        difference_cur_batch += difference
        difference_ratio_cur_batch += difference_ratio


    return difference_cur_batch, difference_ratio_cur_batch


  


def intersection_area(pre_left_up,pre_right_down,gro_left_up,gro_right_down):
    
    x_original,y_original = pre_left_up[0],pre_left_up[1]
   
    x_original_length,y_original_length = pre_right_down[0] - x_original, pre_right_down[1] - y_original

    
    x_other_original,y_other_original = gro_left_up[0],gro_left_up[1]
   
    x_other_original_length, y_other_original_length = gro_right_down[0] - x_other_original, gro_right_down[1] - y_other_original
    

    x_overlap = max(0,min(x_original+x_original_length,x_other_original+x_other_original_length) - max(x_original,x_other_original))
    
 
    y_overlap = max(0,min(y_original+y_original_length,y_other_original+y_other_original_length) - max(y_original,y_other_original))

    return x_overlap * y_overlap


def union_area(pre_left_up,pre_right_down,gro_left_up,gro_right_down):
    
    x_original,y_original = pre_left_up[0],pre_left_up[1]
    x_original_length,y_original_length = pre_right_down[0] - x_original, pre_right_down[1] - y_original

    x_other_original,y_other_original = gro_left_up[0],gro_left_up[1]
    x_other_original_length, y_other_original_length = gro_right_down[0] - x_other_original, gro_right_down[1] - y_other_original

    area1 = x_original_length * y_original_length
    area2 = x_other_original_length * y_other_original_length

    return area1 + area2 - intersection_area(pre_left_up,pre_right_down,gro_left_up,gro_right_down)

def IoU(pre_left_up,pre_right_down,gro_left_up,gro_right_down):
    return intersection_area(pre_left_up,pre_right_down,gro_left_up,gro_right_down) / union_area(pre_left_up,pre_right_down,gro_left_up,gro_right_down)




def IoUCount(prediction,groundTruth):
    
    batchSize = prediction.shape[0]
    allbatchvale = 0.0
    
    for each in range(batchSize):
        cur_pre = prediction[each]
        cur_gro = groundTruth[each]
       
        threshold = 0.0
        cur_pre_index = torch.nonzero(cur_pre>threshold)
        cur_gro_index = torch.nonzero(cur_gro>threshold)
        
        if cur_pre_index.shape[0] < 4:
            iou_vale = 0
        else:
            
            x_min_pre,x_max_pre = torch.min(cur_pre_index[:,0]),torch.max(cur_pre_index[:,0])
            y_min_pre,y_max_pre = torch.min(cur_pre_index[:,1]),torch.max(cur_pre_index[:,1])

            x_min_gro, x_max_gro = torch.min(cur_gro_index[:,0]), torch.max(cur_gro_index[:,0])
            y_min_gro, y_max_gro = torch.min(cur_gro_index[:,1]), torch.max(cur_gro_index[:,1])

            pre_left_up = [x_min_pre,y_min_pre]
            pre_right_down = [x_max_pre,y_max_pre]

            gro_left_up = [x_min_gro,y_min_gro]
            gro_right_down = [x_max_gro,y_max_gro]
            iou_vale = IoU(pre_left_up,pre_right_down,gro_left_up,gro_right_down)

        allbatchvale += iou_vale

    return allbatchvale





