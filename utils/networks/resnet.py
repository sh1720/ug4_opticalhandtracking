import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision.models as models
from config import cfg

class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type=50, pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        # Load the ResNet model from torchvision
        if resnet_type == 18:
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == 34:
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnet_type == 50:
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnet_type == 101:
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnet_type == 152:
            self.resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Invalid ResNet type. Choose from 18, 34, 50, 101, or 152.")
        
        # Remove the fully connected layer to keep only feature extraction
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("ResNet weights initialized")