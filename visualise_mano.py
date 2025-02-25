import torch 
import smplx 
import numpy as np 
import trimesh 
import pyrender 
import matplotlib.pyplot as plt 
import pickle as pickle
from tqdm import tqdm

MANO_MODEL_PATH = "ug4_opticalhandtracking/mano_models/MANO_RIGHT.pkl"
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

# Original pose_params from .pkl file (48 values)
pose_params = torch.tensor(pkl_data['handPose'], dtype=torch.float32).to(device)  # [48]

# Add global orientation (3 values) to match MANO input
# global_orient = torch.zeros(3, dtype=torch.float32).to(device)  # [3]
# pose_params = torch.cat([global_orient, pose_params], dim=0).view(1, -1)  # [1, 51]
shape_params = torch.tensor(pkl_data['handBeta'], dtype=torch.float32).to(device)

pose_params = pose_params.view(1, -1)  # [1, 48] for MANO
shape_params = shape_params.view(1, -1)  # [1, 10] for MANO

# Forward pass through the MANO model
output = mano_layer(
    betas=shape_params,        # Hand shape parameters (identity)
    hand_pose=pose_params[:, 3:],  # Exclude the first 3 values (global orientation)
    global_orient=pose_params[:, :3]  # First 3 values for global orientation
)

# # Extract vertices and joint positions
vertices = output.vertices.detach().cpu().numpy().squeeze()  # [778, 3]
joints = output.joints.detach().cpu().numpy().squeeze()  # [16, 3]

# Print output shape for debugging
print("Vertices shape:", vertices.shape)
print("Joints shape:", joints.shape)

# Convert to trimesh for visualization
mesh = trimesh.Trimesh(vertices, mano_layer.faces, process=False)

# Render using pyrender
scene = pyrender.Scene()

# Convert trimesh to pyrender mesh
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
scene.add(pyrender_mesh)

# Setup camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
scene.add(camera, pose=np.array([
    [1,  0,  0,  0],
    [0,  1,  0,  0],
    [0,  0,  1,  0.5],  # Move camera back slightly
    [0,  0,  0,  1]
]))

# Add lighting
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=np.eye(4))

# Render and display image
r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
color, depth = r.render(scene)

# Display using Matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(color)
plt.axis("off")
plt.title("MANO Hand Model Rendering")
plt.show()

# Close the renderer
r.delete()
