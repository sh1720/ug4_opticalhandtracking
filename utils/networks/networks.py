import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from resnet import ResNetBackbone
from layers import * 

class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)
    
    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        img_feat = self.resnet(img)
        return img_feat

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        
        self.joint_deconv = make_deconv_layers([2048, 256, 256, 256])
        self.joint_conv = make_conv_layers([256, self.joint_num * cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        
        self.root_fc = make_linear_layers([2048, 512, cfg.output_root_hm_shape], relu_final=False)
        self.hand_fc = make_linear_layers([2048, 512, 2], relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(cfg.output_root_hm_shape).float().cuda()[None, :]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, img_feat):
        joint_img_feat = self.joint_deconv(img_feat)
        joint_heatmap3d = self.joint_conv(joint_img_feat).view(-1, self.joint_num, cfg.output_hm_shape[0], cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        
        img_feat_gap = F.avg_pool2d(img_feat, (img_feat.shape[2], img_feat.shape[3])).view(-1, 2048)
        root_heatmap1d = self.root_fc(img_feat_gap)
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1, 1)
        hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))
        
        return joint_heatmap3d, root_depth, hand_type