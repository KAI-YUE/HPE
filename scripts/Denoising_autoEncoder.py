# Python Libraries
import os
import pickle
import random
import time
import numpy as np

# Pytorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class myAutoEncoder(nn.Module):
    def __init__(self, input_size=60, latent_size=20, sigma=10):
        super(myAutoEncoder, self).__init__()
        intermidate_size = int(0.5*input_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, latent_size),
            nn.ReLU()
        )
        
        self.decoder =  nn.Sequential(
            nn.Linear(latent_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size)
        )
        
        # sigma for Gaussian noise
        self.sigma = sigma
        
    def forward(self, x):
        # Draw ranom noise from Gaussian distribution
        # x += self.sigma * torch.randn(x.shape).to(x)
        latent_var = self.encoder(x.view(x.shape[0], -1))
        y = self.decoder(latent_var)
        
        return dict(latent_var=latent_var, y=y) 


class normPoseDataset(Dataset):
    def __init__(self, src_dir):
        """
        Constructor of the Dataset class.
        -------------------------------------------------
        Args
            src_dir :    The directory of the input data.
        """
        with open(src_dir, "rb") as fp:
            X = pickle.load(fp)
                    
        self.X = torch.from_numpy(np.asarray(X, dtype="float32"))
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx]


if __name__ == "__main__":
    
    src_dir = r"F:\DataSets\norm_3d_pos\norm_3d_pos.dat"
    model_dir = r"D:\Projects\HPE\checkpoints\AutoEncoder\myAutoEncoder.pth"
    max_epoch = 1000
    batch_size = 128
    learning_rate = 1e-6
    device = "cuda"
    lambda_ = 0.1
    log_interval = 1
    
    model = myAutoEncoder()
    model = model.to(device)
    optimizer = optim.SGD(
        params = model.parameters(),
        lr = learning_rate
        )
    
    train_data = normPoseDataset(src_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    start = time.time()
    for epoch in range(max_epoch):
        print("---------------------------")
        for iterations, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x = data.to(device)
            output = model(x)
            y = output["y"]
             
            # w_copy = model.encoder[0].weight.data.clone()
            loss = 1/x.shape[0]*torch.sum((x-y)**2) 
            loss.backward()
            optimizer.step()
             
            # # Regularization term of ||Jacobian||
            # reg_term_grad = torch.zeros_like(w_copy)
            # for i in range(batch_size):
            #     w_Jacobian = torch.zeros_like(w_copy)
            #     for j in range(y.shape[0]):
            #         if y[i] != 0:
            #             w_Jacobian[i] = 2*w_copy[i]
            #     reg_term_grad += w_Jacobian 

            # reg_term_grad /= batch_size
            # model.encoder[0].weight.data -= optimizer.lr * lambda_ * reg_term_grad 
            
            if (iterations%log_interval == 0):
                print("epoch{:0d} iter {:0d} loss {:.3f}".format(epoch, iterations, loss))
             
    print("{:.2f} h has elapsed.".format((time.time()-start)/3600))
    torch.save(model.state_dict(), model_dir)
       
        
    
    
    
    
    
    