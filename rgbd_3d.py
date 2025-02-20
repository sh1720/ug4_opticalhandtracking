import numpy as np
import cv2
import open3d as o3d

depthScale = 0.00012498664727900177  # Depth scale from HO3D dataset

depth_image_path = "D:/FYP Datasets/HO3D_v2/train/ABF10/depth/0000.png"  # Replace with your depth image
rgb_image_path = "D:/FYP Datasets/HO3D_v2/train/ABF10/rgb/0000.png"  # Replace with your RGB image

# Load images
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Load depth (single channel)
rgb_image = cv2.imread(rgb_image_path)  # Load RGB image

# Convert RGB from OpenCV BGR to RGB
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

def decodeMultiChannelDepth(depthImg):
    """
    Converts multi-channel depth data (B, G) to a 16-bit single-channel depth in meters.
    """
    dpt = depthImg[:, :, 0] + depthImg[:, :, 1] * 256  # Convert to 16-bit
    dpt = dpt * depthScale  # Convert to meters
    return dpt

depth_meters = decodeMultiChannelDepth(depth_image)

depth_meters[depth_meters == 0] = depthScale  # Replace 0 with a small valid depth (1mm)

# ======= Camera Intrinsics from HO3D =======
camMat = np.array([
    [614.627,   0.   , 320.262],  # fx,  0, cx
    [  0.   , 614.101, 238.469],  #  0, fy, cy
    [  0.   ,   0.   ,   1.   ]   #  0,  0, 1
])

fx, fy = camMat[0, 0], camMat[1, 1]  # Focal lengths
cx, cy = camMat[0, 2], camMat[1, 2]  # Principal points

height, width = depth_meters.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))  # Pixel grid

X = (u - cx) * depth_meters / fx
Y = (v - cy) * depth_meters / fy
Z = depth_meters  # Now all values are valid

# Stack into (N,3) format
points_3d = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize RGB colors to [0,1]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize 3D point cloud
o3d.visualization.draw_geometries([pcd], window_name="HO3D RGB-D to 3D Point Cloud")
