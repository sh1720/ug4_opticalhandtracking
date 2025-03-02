import numpy as np 
import torch 
import torch.utils.data as data
import pickle
import os.path as osp
from preprocessing import load_img, load_skeleton

class Dataset(data.Dataset):   
    def __init__(self, mode, transform = None): 
        self.mode = mode  # Describes if it's test/train/val 
        self.img_path = ''  # Describes image path 
        self.annot_path = ''  # Describes where metadata is stored 
        self.skeleton_path = 'utils/skeleton.txt'

        self.output_path = ''  # Outputs the network path 
        
        self.transform = transform
        self.joint_num = 21 
        self.rl_joint_idx = {'right': 20, 'left': 41}  # Joints 0:20 will be the right hand, 21:41 the left hand 
        self.joint_idx = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(self.skeleton_path, self.joint_num * 2)

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = [] 

        print("Loading annotation from " + self.annot_path)

        with open(self.annot_path, 'rb') as f:
            annot_data = pickle.load(f)
        
        camera_intrinsics = torch.tensor(annot_data['camMat'], dtype=torch.float32)
        pose_params = torch.tensor(annot_data['handPose'], dtype=torch.float32)
        shape_params = torch.tensor(annot_data['handBeta'], dtype=torch.float32)
        hand_joints = torch.tensor(annot_data['handJoints3D'], dtype=torch.float32)
        hand_translation = torch.tensor(annot_data['handTrans'], dtype=torch.float32)
        right_or_left = torch.tensor(1 if annot_data['handType'] == 'right' else 0, dtype=torch.float32)

        # Load and preprocess image
        img_path = osp.join(self.img_path, annot_data['image'])
        image = load_img(img_path)
        
        if self.transform:
            image = self.transform(image)

        return {
            "camera_intrinsics": camera_intrinsics,
            "pose_params": pose_params,
            "shape_params": shape_params,
            "hand_joints": hand_joints,
            "hand_translation": hand_translation,
            "right_or_left": right_or_left,
            "image": image
        }
