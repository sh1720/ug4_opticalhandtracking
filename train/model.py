import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.module import BackboneNet, PoseNet
from nets.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss
from config import cfg
import math

class Model(nn.Module):
    def __init__(self, backbone_net, pose_net):
        super(Model, self).__init__()

        # Modules
        self.backbone_net = backbone_net
        self.pose_net = pose_net
          
        # Loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()
     
    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']
        batch_size = input_img.shape[0]
        img_feat = self.backbone_net(input_img)
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(img_feat)
        
        if mode == 'train':
            loss = {}
            loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, targets['joint_coord'], meta_info['joint_valid'])
            loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, targets['rel_root_depth'], meta_info['root_valid'])
            loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'], meta_info['hand_type_valid'])
            return loss
        elif mode == 'test':
            out = {}
            out['joint_coord'] = joint_heatmap_out
            out['rel_root_depth'] = rel_root_depth_out
            out['hand_type'] = hand_type
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            return out

def init_weights(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def get_model(mode, joint_num):
    backbone_net = BackboneNet()
    pose_net = PoseNet(joint_num)

    if mode == 'train':
        backbone_net.init_weights()
        pose_net.apply(init_weights)

    model = Model(backbone_net, pose_net)
    return model
