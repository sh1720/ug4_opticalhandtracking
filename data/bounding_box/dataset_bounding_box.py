# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import scipy.io
from PIL import Image
import cv2
import random
from support_function import getListBoxes, getBoxes, convertBoxes  # Ensure these functions are properly defined

class EgoDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the EgoHands dataset.
    """

    def __init__(self, path, n_elements=-1, transforms=None):
        """
        Initializes the dataset.

        Args:
            path (str): Path to the dataset directory.
            n_elements (int): Number of elements to load (-1 loads all).
            transforms: Transformations to apply to images.
        """
        if n_elements > 2400:
            n_elements = 450  # Limiting to a maximum of 450 images

        num_folders = n_elements // 100  # Number of full folders to use
        self.transforms = transforms
        self.get_image_PIL = False

        # Retrieve all folder paths
        folder_list = [element for element in os.walk(path)][1:]  # Skip root folder
        folder_list = random.sample(folder_list, num_folders + 1)

        self.elements = []
        self.boxes_list = []

        for folder in folder_list:
            folder_path = folder[0]
            file_list = sorted(folder[2])  # Sort to maintain consistency

            # Load bounding boxes from the annotation file
            boxes = np.squeeze(scipy.io.loadmat(os.path.join(folder_path, file_list[100]))['polygons'])
            boxes = getListBoxes(boxes)

            # Store image paths and corresponding bounding boxes
            for i, photo_name in enumerate(file_list[:100]):  
                self.elements.append(os.path.join(folder_path, photo_name))
                self.boxes_list.append(getBoxes(i, boxes))

        # Shuffle dataset
        indices = np.arange(len(self.elements))
        np.random.shuffle(indices)
        self.elements = [self.elements[i] for i in indices]
        self.boxes_list = [self.boxes_list[i] for i in indices]

    def __getitem__(self, idx):
        """
        Retrieves an image and its target labels.

        Args:
            idx (int): Index of the dataset.

        Returns:
            tuple: (image, target)
        """
        # Load image
        img = Image.open(self.elements[idx]).convert("RGB")

        # Load and convert bounding boxes
        boxes = convertBoxes(self.boxes_list[idx])
        self.last_boxes = boxes
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Prepare target labels for RCNN
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply transformations if specified
        if self.transforms and not self.get_image_PIL:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.elements)

    def checkBoxes(self, idx, color=(255, 0, 128), thickness=2):
        """
        Displays an image with bounding boxes using OpenCV.

        Args:
            idx (int): Index of the image.
            color (tuple): Color of the bounding boxes.
            thickness (int): Thickness of bounding box lines.

        Returns:
            tuple: (image, bounding boxes)
        """
        prev_get_image_PIL = self.get_image_PIL
        self.get_image_PIL = True
        img, target = self[idx]
        self.get_image_PIL = prev_get_image_PIL

        img_opencv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = target['boxes'].numpy()

        for box in boxes:
            cv2.rectangle(img_opencv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=thickness)

        cv2.imshow(f"Check Box {idx} - {self.elements[idx]}", img_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img, boxes

    def checkBoxes_2(self, idx, color=(255, 0, 128), thickness=2):
        """
        Displays bounding boxes using an alternative method.

        Args:
            idx (int): Index of the image.
            color (tuple): Color of the bounding boxes.
            thickness (int): Thickness of bounding box lines.
        """
        img_opencv = cv2.imread(self.elements[idx])
        boxes = self.boxes_list[idx]

        for box in boxes:
            cv2.rectangle(img_opencv, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), color=color, thickness=thickness)

        cv2.imshow(f"Check Box {idx} - {self.elements[idx]}", img_opencv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_transform(train=True):
    """
    Returns a torchvision transformation pipeline.

    Args:
        train (bool): Whether to apply training augmentations.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_transform_2():
    """
    Alternative transformation pipeline with OpenCV preprocessing.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    return T.Compose([TransformOpenCV(), T.ToTensor()])

class TransformOpenCV(object):
    """
    Custom OpenCV-based transformation class.
    """

    def __call__(self, img):
        kernel = np.ones((4, 4), np.uint8)
        
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.inRange(img, np.array([2, 0, 0]), np.array([20, int(255 * 0.68), 255]))
        img = cv2.dilate(img, kernel, iterations=2)
        img = cv2.erode(img, kernel, iterations=2)

        return img
