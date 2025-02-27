import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from tqdm import tqdm 

class EgoHandsDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = sorted(os.listdir(img_dir))
        self.labels = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                x_center, y_center, width, height = map(float, parts[1:])
                x_min = (x_center - width / 2) * w
                y_min = (y_center - height / 2) * h
                x_max = (x_center + width / 2) * w
                y_max = (y_center + height / 2) * h
                boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # 1 = hand

        target = {"boxes": boxes, "labels": labels}
        img = T.ToTensor()(img)

        return img, target

dataset = EgoHandsDataset("D:/FYP Datasets/egohands_preprocessed/images", "D:/FYP Datasets/egohands_preprocessed/labels")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # Background + Hand
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    
    epoch_loss = 0  # Track total loss for the epoch
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
        for imgs, targets in progress_bar:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())  # Show loss in progress bar
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {epoch_loss / len(dataloader):.4f}")

