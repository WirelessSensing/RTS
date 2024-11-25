#encoding=utf-8

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RTIDataset(Dataset):
    def __init__(self,fileName) -> None:
        self.filenameList = []
        self.groundList = []
        with open(fileName,"r",encoding="utf-8") as f:
            flines = f.readlines()
            for line in flines:
                line = line.strip().split("\t")
                self.filenameList.append(line[0])
                self.groundList.append(line[1])
                
    def __getitem__(self, index):
        return torch.load(self.filenameList[index]).float(),torch.from_numpy(np.load(self.groundList[index])).float()
               
    def __len__(self):
        return len(self.filenameList)
    


