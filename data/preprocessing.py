import os
import cv2
import numpy as np
# from config import cfg
import random
import math

def load_img(path, order='RGB'):

    image = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(image, np.ndarray):
        raise IOError("Fail to read image path", path)

    if order=='RGB':
        image = image[:,:,::-1].copy() 

    image = image.astype(np.float32)
    return image 


def load_skeleton(path, joint_num):

    skeleton = [{} for _ in range(joint_num)]

    with open(path) as path: 
        for line in path: 
            if line.startswith('#'): continue
            splitted = line.strip().split()
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id

    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i: 
                joint_child_id.append(i)
        skeleton[i]['child_id'] = joint_child_id

    return skeleton 