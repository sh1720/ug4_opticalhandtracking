import torch
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from interhand_model import InterHandModel  # Assuming you have an interhand_model.py file

# ✅ Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load InterHand2.6M model
model = InterHandModel()  # Custom class for InterHand2.6M
model.load_state_dict(torch.load("C:/Users/sreed/Downloads/interhand_checkpoint.pth", map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully!")

def draw_keypoints(img, keypoints, color=(0, 255, 0)):
    """
    Draw 3D hand keypoints on the image.
    """
    img = np.ascontiguousarray(img)  # Ensure compatibility with OpenCV

    for point in keypoints:
        x, y, z = map(int, point[:3])  # Extract (x, y, z) coordinates
        cv2.circle(img, (x, y), 5, color, -1)  # Draw keypoints

    return img

def predict_and_visualize(image_path):
    """
    Loads an image, performs 3D hand pose estimation, and visualizes keypoints.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img_rgb).to(device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    pred_keypoints = outputs[0]["joints_3d"].cpu().numpy()  # Extract 3D keypoints

    img = draw_keypoints(img, pred_keypoints, (0, 255, 0))  # Green = Keypoints

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# ✅ Test on a sample image
predict_and_visualize("D:/FYP Datasets/egohands_preprocessed/images/CARDS_COURTYARD_T_B_16.jpg")  # Update your path
