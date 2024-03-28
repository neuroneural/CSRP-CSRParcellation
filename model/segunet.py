
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# segmentation U-Net
class SegUnet(nn.Module):
    def __init__(self, c_in=1, c_out=2):
        super(SegUnet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=c_in, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3,
                               stride=2, padding=1)
        self.conv5 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3,
                               stride=2, padding=1)

        self.deconv4 = nn.Conv3d(in_channels=256, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.deconv3 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.deconv2 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.deconv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        
        self.lastconv1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3,
                                   stride=1, padding=1)
        self.lastconv2 = nn.Conv3d(in_channels=16, out_channels=c_out, kernel_size=3,
                                   stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self, x):

        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x  = F.leaky_relu(self.conv5(x4), 0.2)
        x  = self.up(x)
        
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = self.up(x)
        
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)

        x = F.leaky_relu(self.lastconv1(x), 0.2)
        x = self.lastconv2(x)

        return x