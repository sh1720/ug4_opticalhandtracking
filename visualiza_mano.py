import torch 
import smplx 
import numpy as np 
import trimesh 
import pyrender 
import matplotlib.pyplot as plt 
import pickle.mixin as pickle
from tqdm import tqdm

MANO_MODEL_PATH = "D:/FYP Datasets/mano_v1_2/models/MANO_RIGHT.pkl"
pkl_file_path = "D:/FYP Datasets/HO3D_v2/train/ABF10/meta/0000.pkl"

# Load the .pkl file with a progress bar
with open(pkl_file_path, 'rb') as f:
    pkl_bytes = f.read()  # Read entire file with progress
    for _ in tqdm(range(len(pkl_bytes)), desc="Loading .pkl", unit="byte"):
        pass
    pkl_data = pickle.loads(pkl_bytes)  # Deserialize data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mano_layer = smplx.MANO(
    model_path = MANO_MODEL_PATH,
    is_rhand = True, 
    use_pca = False, 
    flat_hand_mean = True
).to(device)

pose_params = torch.tensor(pkl_data['handPose'], dtype=torch.float32).to(device)
shape_params = torch.tensor(pkl_data['handBeta'], dtype=torch.float32).to(device)

pose_params = pose_params.view(1, -1)  # [1, 48] for MANO
shape_params = shape_params.view(1, -1)  # [1, 10] for MANO

output = mano_layer(
    global_orient=torch.zeros(1,3).to(device),
    hand_pose = pose_params, 
    betas = shape_params, 
)

vertices = output.vertices.detach().cpu().numpy().squeeze()
joints = output.joints.detach().cpu().numpy().squeeze()

