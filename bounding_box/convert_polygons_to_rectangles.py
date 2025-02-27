import os
import cv2
import numpy as np
import scipy.io

# Path to the EgoHands dataset
data_path = "D:/FYP Datasets/egohands_data/_LABELLED_SAMPLES"
output_images = "D:/FYP Datasets/egohands_preprocessed/images"
output_labels = "D:/FYP Datasets/egohands_preprocessed/labels"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Go through each folder in the dataset
for folder in os.listdir(data_path):
    mat_file = os.path.join(data_path, folder, "polygons.mat")
    if not os.path.exists(mat_file):
        continue

    # Load segmentation masks
    mat = scipy.io.loadmat(mat_file)
    polygons = mat["polygons"][0]  # Each frame has 4 hands

    for i, poly_set in enumerate(polygons):  # Iterate over frames
        img_path = os.path.join(data_path, folder, f"frame_{i:04d}.jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        boxes = []
        for poly in poly_set:
            if poly.size == 0:
                continue
            x_min, y_min = np.min(poly, axis=0)
            x_max, y_max = np.max(poly, axis=0)
            boxes.append([x_min, y_min, x_max, y_max])

        # Save bounding boxes as text files (YOLO format)
        label_file = os.path.join(output_labels, f"{folder}_{i}.txt")
        with open(label_file, "w") as f:
            for box in boxes:
                x_center = (box[0] + box[2]) / (2 * w)
                y_center = (box[1] + box[3]) / (2 * h)
                width = (box[2] - box[0]) / w
                height = (box[3] - box[1]) / h
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

        # Save processed images
        cv2.imwrite(os.path.join(output_images, f"{folder}_{i}.jpg"), img)

print("Dataset is ready!")
