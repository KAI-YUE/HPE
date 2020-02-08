# Python Libraries
import os
import cv2
import pickle
import numpy as np

# Pytorch Libraries
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# My Libraries

class HLoDataset(Dataset):
    def __init__(self, data_dir):
        """
        Constructor of the Dataset class for HolNet.
        -------------------------------------------------
        Args
            data_dir :    The directory of the input data.
        """
        self.ImageList = []
        
        for root, dirs, files in os.walk(data_dir):
            if (files != []):
                for f in files:
                    if "dat" in f:
                        self.ImageList.append(os.path.join(root, f))
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        
    def __len__(self):
        return len(self.ImageList)
    
    def __getitem__(self, idx):
        with open(self.ImageList[idx], 'rb') as fp:
            a_set = pickle.load(fp) 

        Img = a_set["img"].transpose((2, 0, 1))
        Img = torch.from_numpy(Img).to(torch.float32)
        Img = (Img - self.mean)/self.std

        wrist = cv2.resize(a_set["W_hm"].astype('float32'), (Img.shape[2], Img.shape[1]))
        wrist = torch.from_numpy(wrist)
        wrist = wrist.view(-1, wrist.shape[0], wrist.shape[1])

        return dict(img=Img, w=wrist)

    def create_iterator(self, batch_size=4):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class PReDataset(Dataset):
    def __init__(self, data_dir):
        """
        Constructor of the Dataset class for PReNet.
        -------------------------------------------------
        Args
            data_dir :    The directory of the input data.
        """
        self.ImageList = []

        for root, dirs, files in os.walk(data_dir):
            if (files != []):
                for f in files:
                    if "dat" in f:
                        self.ImageList.append(os.path.join(root, f))

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.ImageList)

    def __getitem__(self, idx):
        with open(self.ImageList[idx], 'rb') as fp:
            a_set = pickle.load(fp) 

        ROI = a_set["ROI"]
        Img = a_set["img"][ROI[0]:ROI[1], ROI[2]:ROI[3]].transpose((2, 0, 1))
        Img = torch.from_numpy(Img.astype('float32'))
        Img = (Img - self.mean)/self.std
        
        # Heatmaps of different parts
        hm = torch.from_numpy(a_set["heatmaps"]).to(torch.float32)
        
        pos = torch.from_numpy(a_set['pos_arr'].astype('float32')).view(
                a_set['pos_arr'].shape[0], a_set['pos_arr'].shape[1], 1)

        return dict(img=Img, hm=hm, pos=pos)

    def create_iterator(self, batch_size=4):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


