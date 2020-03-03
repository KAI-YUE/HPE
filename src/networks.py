# Pytorch Libraries
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

        self.fc_theta1 = nn.Linear(256*16**2, 256)
        self.fc_theta2 = nn.Linear(256, 33)
        
        self.fc_scale1 = nn.Linear(256*16**2, 256)
        self.fc_scale2 = nn.Linear(256, 20)

        self.phalanges_arr =  torch.tensor([40.712, 34.040, 29.417, 26.423,
                                            79.706, 35.224, 23.270, 22.036,
                                            75.669, 41.975, 26.329, 24.504,
                                            75.358, 39.978, 23.513, 22.647,
                                            74.556, 27.541, 19.826, 20.395])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)

        x = self.res3a(x)
        x = self.res3b(x)
        x = self.res3c(x)

        x = self.res4a(x)
        x = self.res4b(x)
        x = self.res4c(x)

        x = self.conv4e(x)
        x = self.conv4f(x)

        theta = self.fc_theta1(x.view(x.shape[0], -1))
        theta = 3.14*torch.sigmoid(self.fc_theta2(F.relu(theta)))

        scale = self.fc_scale1(x.view(x.shape[0], -1))
        scale = torch.sigmoid(self.fc_scale2(F.relu(scale))) 
        pos = self.forward_kinematics(scale, theta)

        return dict(pos=pos, scale=scale, theta=theta)


    def forward_kinematics(self, scale, theta):
        """
        Derive the 3d positions with forward kinematics.
        """
        scale = scale + 1
        pos = torch.zeros((theta.shape[0], 21, 3))
        p0 = torch.tensor([0., 0., 0., 1.])

        # For each sample
        
        for i in range(theta.shape[0]):
            j = 0
            # Thumb kinematics
            z = self.z_matrix(theta[i,0])
            x = self.x_matrix(theta[i,1], torch.tensor(0.))
            T = z @ x

            z = self.z_matrix(theta[i,2])
            x = self.x_matrix(theta[i,3], scale[i, j]*self.phalanges_arr[j])
            j += 1
            T = T @ z @ x
            pos[i,1] = (T @ p0) [:3]

            z = self.z_matrix(theta[i,4])
            x = self.x_matrix(theta[i,5], torch.tensor(0.))
            T = T @ z @ x

            z = self.z_matrix(theta[i,6])
            x = self.x_matrix(torch.tensor(0.), scale[i, j]*self.phalanges_arr[j])
            j += 1
            T = T @ z @ x
            pos[i,2] = (T @ p0) [:3]

            z = self.z_matrix(theta[i,7])
            x = self.x_matrix(torch.tensor(0.), scale[i, j]*self.phalanges_arr[j])
            j += 1
            T = T @ z @ x
            pos[i,3] = (T @ p0) [:3]

            z = self.z_matrix(theta[i,8])
            x = self.x_matrix(torch.tensor(0.), scale[i, j]*self.phalanges_arr[j])
            j += 1
            T = T @ z @ x
            pos[i,4] = (T @ p0) [:3]

            # Finger kinematics
            for k in range(4):
                z = self.z_matrix(theta[i,6*k+9])
                x = self.x_matrix(torch.tensor(0.), scale[i, j]*self.phalanges_arr[j])
                j += 1
                T = z @ x
                pos[i,4*k+5] = (T @ p0) [:3]

                z = self.z_matrix(theta[i,6*k+10])
                x = self.x_matrix(theta[i,6*k+11], torch.tensor(0.))
                T = T @ z @ x

                for l in range(3):
                    z = self.z_matrix(theta[i,6*k+12+l])
                    x = self.x_matrix(torch.tensor(0.), scale[i, j]*self.phalanges_arr[j])
                    j += 1
                    T = T @ z @ x
                    pos[i,4*k+6+l] = (T @ p0) [:3]

        return pos[:,None,...]


    @ staticmethod
    def z_matrix(theta, d=0):
        """
        Return the z matrix given the D-H parameter.
        To guarantee the gradient calculation, we assign the value manually.
        """
        z = torch.eye(4)
        z[0][0] = torch.cos(theta)
        z[0][1] = -torch.sin(theta)
        z[1][0] = torch.sin(theta)
        z[1][1] = torch.cos(theta)
        z[2][3] = d
        return z

    @ staticmethod
    def x_matrix(alpha, a=0):
        """
        Return the x matrix given the D-H parameter.
        To guarantee the gradient calculation, we assign the value manually.
        """
        x = torch.eye(4)
        x[0][3] = a
        x[1][1] = torch.cos(alpha)
        x[1][2] = -torch.sin(alpha)
        x[2][1] = torch.sin(alpha)
        x[2][2] = torch.cos(alpha)
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