# Python Libraries
import os
import cv2
import pickle
import numpy as np

# Pytorch Libraries
import torch
import torch.nn.functional as F
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
                
    def __len__(self):
        return len(self.ImageList)
    
    def __getitem__(self, idx):
        with open(self.ImageList[idx], 'rb') as fp:
            a_set = pickle.load(fp) 

        Img = np.dstack((a_set["depth_norm"], a_set["img"]))
        Img = Img.transpose((2,0,1))
        Img = torch.from_numpy(Img).to(torch.float32)

        hm = a_set["root_hm"].astype('float32')
        hm = torch.from_numpy(hm)[None,...]

        return dict(img=Img, hm=hm)

    def create_iterator(self, batch_size=1):
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
        self.img_size = [128, 128]

        for root, dirs, files in os.walk(data_dir):
            if (files != []):
                for f in files:
                    if "dat" in f:
                        self.ImageList.append(os.path.join(root, f))

    def __len__(self):
        return len(self.ImageList)

    def __getitem__(self, idx):

        with open(self.ImageList[idx], 'rb') as fp:
            a_set = pickle.load(fp) 

        # ROI = a_set["ROI"].astype("int")
        # img = a_set["img"][ROI[0]:ROI[1], ROI[2]:ROI[3]]
        # depth = a_set["depth_norm"][ROI[0]:ROI[1], ROI[2]:ROI[3]]
        # Img = np.dstack((depth, img)).transpose((2,0,1))
        # Img = torch.from_numpy(Img).to(torch.float32)
        # Img = F.interpolate(Img[None, ...], self.img_size, mode="bilinear")[0]
        img = a_set["cropped_img"]
        depth = a_set["cropped_depth"]
        Img = torch.from_numpy(np.dstack((depth, img)).transpose((2,0,1))).to(torch.float32)
        
        # Heatmaps of different parts
        hm = torch.from_numpy(a_set["heatmaps"]).to(torch.float32)
        
        pos = torch.from_numpy(a_set['norm_3d_pos'].astype('float32')).view(1,21,3)

        return dict(img=Img, hm=hm, pos=pos)
            

    def create_iterator(self, batch_size=1):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


