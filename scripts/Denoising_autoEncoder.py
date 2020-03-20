# Python Libraries
import os
import pickle
import random
import numpy as np

# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DAE_2L(nn.Module):
    def __init__(self, input_size=60, latent_size=20, intermidate_size=40, sigma=0.0001):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, intermidate_size),
            nn.ReLU(),
            nn.Linear(intermidate_size, latent_size),
            )
        
        self.decoder =  nn.Sequential(
            nn.Linear(latent_size, intermidate_size),
            nn.ReLU(),
            nn.Linear(intermidate_size, input_size),
            )
        
        # sigma for Gaussian noise
        self.sigma = sigma
        
    def forward(self, x):
        # Draw ranom noise from Gaussian distribution
        x += self.sigma * torch.randn(x.shape).to(x)
        latent_var = self.encoder(x.view(x.shape[0], -1))
        y = self.decoder(latent_var)
        
        return dict(latent_var=latent_var, y=y) 


class DAE_1L(nn.Module):
    def __init__(self, input_size=60, latent_size=1000, sigma=0.001):
        super(DAE_1L, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            )
        
        self.decoder =  nn.Sequential(
            nn.Linear(latent_size, input_size),
            )
        
        # sigma for Gaussian noise
        self.sigma = sigma
        
    def forward(self, x):
        # Draw ranom noise from Gaussian distribution
        x += self.sigma * torch.randn(x.shape).to(x)
        latent_var = self.encoder(x.view(x.shape[0], -1))
        y = self.decoder(latent_var)
        
        return dict(latent_var=latent_var, y=y) 

class normPoseDataset(Dataset):
    def __init__(self, X):
        """
        Constructor of the Dataset class.
        ------------------------------------
        Args
            src_dir :    The directory of the input data.
        """
        
        self.X = torch.from_numpy(np.asarray(X, dtype="float32"))
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx]


if __name__ == "__main__":

    dat_dir = r"D:\YUE\norm_3d_pos\kinematics.dat"
    model_dir = r"D:\Projects\DAE_kinematics_2L_29_29_12.pth"

#    dat_dir = r"D:\YUE\norm_3d_pos\theta.dat"
#    model_dir = r"D:\Projects\scale_2L_29_7.pth"
    
    max_epoch = 200
    batch_size = 128
    learning_rate = 1e-4
    device = "cpu"
    lambda_ = 0.1
    
    model = DAE_2L(29, 12, 20, sigma=0.01)
#    model = DAE_2L(60, 40, 40, sigma=0.0001)
#    model = DAE_3L(29, 7, 29, sigma=0.01)
    model = model.to(device)
    optimizer = optim.Adam(
        params = model.parameters(),
        lr = learning_rate,
        weight_decay=0.00005)
    
    with open(dat_dir, "rb") as fp:
        a_set = pickle.load(fp)
    
    X = a_set["theta"]
    
    train_data = normPoseDataset(X)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(max_epoch):
         for iterations, data in enumerate(train_loader, 0):
             optimizer.zero_grad()
             x = data.to(device)
             output = model(x)
             y = output["y"]
             
             loss = torch.sum((x-y)**2) 
             loss.backward()
             optimizer.step()
             
             
             print("epoch {} iter {:0d} loss {:.5f}".format(epoch, iterations, loss))
             
    
    torch.save(model.state_dict(), model_dir)
    

    
       
        
    
    
    
    
    
    