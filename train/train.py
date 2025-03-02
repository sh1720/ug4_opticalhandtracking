import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from tqdm import tqdm 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main(): 
    args = parse_args() 
    cdg.set_args(args.gpu_ids, args.continue_train) #!!
    cudnn.benchmark = True 

    trainer = Trainer() 
    trainer._make
    