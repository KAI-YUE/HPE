# Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class HALNet(nn.Module):
    def __init__(self, in_dim=3):
        super(HALNet, self).__init__()

        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res2a = Conv_ResnetBlock(64, 64, 256)
        self.res2b = Skip_ResnetBlock(256, 64, 256)
        self.res2c = Skip_ResnetBlock(256, 64, 256)

        self.res3a = Conv_ResnetBlock(256, 128, 512, stride=2)
        self.res3b = Skip_ResnetBlock(512, 128, 512)
        self.res3c = Skip_ResnetBlock(512, 128, 512)

        self.res4a = Conv_ResnetBlock(512, 256, 1024, stride=2)
        self.res4b = Skip_ResnetBlock(1024, 256, 1024)
        self.res4c = Skip_ResnetBlock(1024, 256, 1024)
        self.res4d = Skip_ResnetBlock(1024, 256, 1024)

        self.conv4e = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv4f = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # Intermediate convolution layers for loss calculation 
        self.inter_conv1 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.inter_conv2 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.inter_conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.main_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1) 
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)

        x = self.res3a(x)
        interm1 = self.inter_conv1(x) 
        x = self.res3b(x)
        x = self.res3c(x)

        x = self.res4a(x)
        interm2 = self.inter_conv2(x)
        x = self.res4b(x)
        x = self.res4c(x)

        x = self.conv4e(x)
        interm3 = self.inter_conv3(x)
        x = self.conv4f(x)
        y = self.main_conv(x)

        return y, [interm1, interm2, interm3]

class JORNet(nn.Module):
    def __init__(self, in_dim=3):
        super(JORNet, self).__init__()

        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.res2a = Conv_ResnetBlock(64, 64, 256)
        self.res2b = Skip_ResnetBlock(256, 64, 256)
        self.res2c = Skip_ResnetBlock(256, 64, 256)

        self.res3a = Conv_ResnetBlock(256, 128, 512, stride=2)
        self.res3b = Skip_ResnetBlock(512, 128, 512)
        self.res3c = Skip_ResnetBlock(512, 128, 512)

        self.res4a = Conv_ResnetBlock(512, 256, 1024, stride=2)
        self.res4b = Skip_ResnetBlock(1024, 256, 1024)
        self.res4c = Skip_ResnetBlock(1024, 256, 1024)
        self.res4d = Skip_ResnetBlock(1024, 256, 1024)

        self.conv4e = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv4f = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # Intermediate convolution layers for loss calculation 
        self.inter_conv1 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=(12, 32), stride=10)
        self.inter_conv2 = nn.Conv2d(in_channels=1024, out_channels=21, kernel_size=(5, 16), stride=4)
        self.inter_conv3 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=(5, 16), stride=4)
        
        self.jo_conv = nn.Conv2d(in_channels=256, out_channels=21, kernel_size=(5, 16), stride=4)
        self.hm_conv = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.hm_deconv = nn.ConvTranspose2d(in_channels=64, out_channels=21, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)

        x = self.res3a(x)
        interm1 = self.inter_conv1(x) 
        x = self.res3b(x)
        x = self.res3c(x)

        x = self.res4a(x)
        interm2 = self.inter_conv2(x)
        x = self.res4b(x)
        x = self.res4c(x)

        x = self.conv4e(x)
        interm3 = self.inter_conv3(x)
        x = self.conv4f(x)

        jo_y = self.jo_conv(x)

        x = self.hm_conv(x)
        hm_y = self.hm_deconv(x)

        return [hm_y, jo_y], [interm1, interm2, interm3]

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


class Skip_ResnetBlock(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim, stride=1):
        super(Skip_ResnetBlock, self).__init__()

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

    def forward(self, x):
        y = x + self.basic_block(x)
        return F.relu(y)
