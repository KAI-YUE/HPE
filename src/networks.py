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

class JLoNet(nn.Module):
    def __init__(self, in_dim=4):
        super(JLoNet, self).__init__()

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

        # Output heatmaps 
        self.Conv_hm = Conv_ResnetBlock(128, 64, 21, stride=1) 

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

        y = self.Conv_hm(x)

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
        self.Conv_hm = Conv_ResnetBlock(128, 64, 21, stride=1) 
        
        self.Conv_pos = Conv_ResnetBlock(128, 128, 64, stride=2)
        self.fc1 = nn.Linear(64**3, 512)
        self.fc2 = nn.Linear(512, 21*3)

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

        heatmaps = self.Conv_hm(x)

        pos = self.Conv_pos(x)
        pos = self.fc1(pos.view(pos.shape[0], -1))
        pos = self.fc2(F.relu(pos))
        pos = pos.view(pos.shape[0], 1, 21,3)
        
        return dict(heatmaps=heatmaps, pos=pos)


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


class DAE_2L(nn.Module):
    def __init__(self, input_size=60, latent_size=20, intermidate_size=40, sigma=0.005):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, intermidate_size),
            nn.ReLU(),
            nn.Linear(intermidate_size, latent_size),
            )
        
        self.decoder =  nn.Sequential(
            nn.Linear(latent_size, intermidate_size),
            nn.ReLU(),
            nn.Linear(intermidate_size, input_size)
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
    def __init__(self, input_size=60, latent_size=1000, sigma=0.005):
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
