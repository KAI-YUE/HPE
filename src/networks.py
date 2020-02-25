﻿# Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class HLoNet(nn.Module):
    def __init__(self, in_dim=4):
        super(HLoNet, self).__init__()

        self.Conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder part
        self.ConvB1 = Conv_ResnetBlock(64, 64, 128, stride=1)
        self.ConvB2 = Conv_ResnetBlock(128, 128, 256, stride=2)
        self.ConvB3 = Conv_ResnetBlock(256, 256, 512, stride=2)
        self.ConvB4 = Conv_ResnetBlock(512, 512, 1024, stride=2)

        # Decoder part
        self.ConvT1 = ConvTransBlock(1024, 1024, kernel=4, stride=2, padding=1)
        self.ConvT2 = ConvTransBlock(1024+512, 512, kernel=4, stride=2, padding=1)
        self.ConvT3 = ConvTransBlock(512+256, 256, kernel=4, stride=2, padding=1)
        self.ConvT4 = ConvTransBlock(256+128, 128, kernel=4, stride=2, padding=1)

        # Output conv
        self.Conv2 = Conv_ResnetBlock(128, 64, 1, stride=1) 

    def forward(self, x):
        x = self.Conv1(x)
        x = self.maxpool(x)

        x1 = self.ConvB1(x)
        x2 = self.ConvB2(x1)
        x3 = self.ConvB3(x2)
        x4 = self.ConvB4(x3)

        x = self.ConvT1(x4)
        x = self.ConvT2(torch.cat((x,x3), dim=1))
        x = self.ConvT3(torch.cat((x,x2), dim=1))
        x = self.ConvT4(torch.cat((x,x1), dim=1))

        y = self.Conv2(x)

        # return 2*torch.sigmoid(y) - 1
        return y


class PReNet(nn.Module):
    def __init__(self, in_dim=4):
        super(PReNet, self).__init__()

        self.Conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Encoder part
        self.ConvB1 = Conv_ResnetBlock(64, 64, 128, stride=1)
        self.ConvB2 = Conv_ResnetBlock(128, 128, 256, stride=2)
        self.ConvB3 = Conv_ResnetBlock(256, 256, 512, stride=2)
        self.ConvB4 = Conv_ResnetBlock(512, 512, 1024, stride=2)

        # Decoder part
        self.ConvT1 = ConvTransBlock(1024, 1024, kernel=4, stride=2, padding=1)
        self.ConvT2 = ConvTransBlock(1024+512, 512, kernel=4, stride=2, padding=1)
        self.ConvT3 = ConvTransBlock(512+256, 256, kernel=4, stride=2, padding=1)
        self.ConvT4 = ConvTransBlock(256+128, 128, kernel=3, stride=1, padding=1)

        # Output heatmaps and joint pos
        # self.Conv_hm = Conv_ResnetBlock(128, 64, 21, stride=1) 
        
        self.Conv_Pos1 = Conv_ResnetBlock(128, 128, 64, stride=2)
        self.Conv_Pos2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc_theta1 = nn.Linear(64*32**2, 256)
        self.fc_theta2 = nn.Linear(256, 23)
        
        self.fc_scale1 = nn.Linear(64*32**2, 256)
        self.fc_scale2 = nn.Linear(256, 20)

        self.phalanges_arr = \ 
            torch.tensor([40.712, 34.040, 29.417, 26.423,
                          79.706, 35.224, 23.270, 22.036,
                          75.669, 41.975, 26.329, 24.504,
                          75.358, 39.978, 23.513, 22.647,
                          74.556, 27.541, 19.826, 20.395])

    def forward_kinematics(self, scale, theta):
        """
        Derive the 3d positions with forward kinematics.
        """
        pos = torch.zeros((theta.shape[0], 21, 3))
        p0 = torch.tensor([0., 0., 0., 1.])

        # For each sample
        j = 0
        for i in range(theta.shape[0]):
            # Thumb kinematics
            z = self.z_matrix(theta[i,0])
            x = self.x_matrix(theta[i,1], scale[j]*self.phalanges_arr[j])
            j += 1
            T = z @ x
            pos[i,0] = T @ p0 [:3]

            z = self.z_matrix(theta[i,2])
            x = self.x_matrix(theta[i,3], scale[j]*self.phalanges_arr[j])
            j += 1
            T = T @ z @ x
            pos[i,1] = T @ p0 [:3]

            z = self.z_matrix(theta[i,4])
            x = self.x_matrix(0, scale[j]*self.phalanges_arr[j])
            j += 1
            T = T @ z @ x
            pos[i,2] = T @ p0 [:3]

            z = self.z_matrix(theta[i,5])
            x = self.x_matrix(0, scale[j]*self.phalanges_arr[j])
            j += 1
            T = T @ z @ x
            pos[i,3] = T @ p0 [:3]

            # Finger kinematics
            T = torch.identity(4)
            for k in range(4):
                z = self.z_matrix(theta[i,k+5])
                x = self.x_matrix(theta[i,k+6], scale[j]*self.phalanges_arr[j])
                j += 1
                T = z @ x
                pos[i,4] = T @ p0 [:3]



    def forward(self, x):
        x = self.Conv1(x)
        x = self.maxpool(x)

        x1 = self.ConvB1(x)
        x2 = self.ConvB2(x1)
        x3 = self.ConvB3(x2)
        x4 = self.ConvB4(x3)

        x = self.ConvT1(x4)
        x = self.ConvT2(torch.cat((x,x3), dim=1))
        x = self.ConvT3(torch.cat((x,x2), dim=1))
        x = self.ConvT4(torch.cat((x,x1), dim=1))

        # hm = self.Conv_hm(x)
        hm = 0

        x = self.Conv_theta1(x)
        x = self.Conv_theta2(x)
        
        theta = self.fc_theta1(x.view(x.shape[0], -1))
        theta = 3.14*F.sigmoid(self.fc_theta2(F.relu(theta)))

        scale = self.fc_scale1(x.view(x.shape[0], -1))
        scale = F.relu(self.fc_scale2(F.relu(scale)))
        pos = self.forward_kinematics(scale, theta)
        
        return [hm, pos]

    @ staticmethod
    def z_matrix(self, theta, d=0):
        """
        Return the z matrix given the D-H parameter.
        """
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0, 0],
                              [np.sin(theta), np.cos(theta), 0, 0],
                              [0, 0, 1, d],
                              [0, 0, 0, 1]])

    def x_matrix(self, alpha, a=0):
        """
        Return the x matrix given the D-H parameter.
        """
        return torch.tensor([[1, 0, 0, a],
                            [0, np.cos(alpha), -np.sin(alpha), 0],
                            [0, np.sin(alpha), np.cos(alpha), 0],
                            [0, 0, 0, 1]])

class Regressor(nn.Module):
    """
    Regressor for 3d joint positions.
    """
    def __init__(self, in_dim, out_dim):
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128**3, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.fc3 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.batch_norm(x)
        
        x = F.relu(self.conv1(x))
        x = F.batch_norm(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Conv_ResnetBlock(nn.Module):
    """
    Resnet Block.
    """
    def __init__(self, in_dim, inter_dim, out_dim, stride=1):
        super(Conv_ResnetBlock, self).__init__()

        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(),

            nn.Conv2d(in_channels=inter_dim, out_channels=inter_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(),

            nn.Conv2d(in_channels=inter_dim, out_channels=out_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_dim)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        y = self.basic_block(x) + self.conv_block(x)
        return F.relu(y)

class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(ConvTransBlock, self).__init__()
        
        self.convT_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.convT_block(x)


def init_weights(module, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(module.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight.data, gain=gain)

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)
