#!/usr/bin/python
# -*- encoding: utf-8 -*-

# reference repo: https://github.com/bat67/pytorch-FCN-easiest-demo/blob/master/BagData.py


import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class FaceDataLoader(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        imgA_name = os.listdir(self.image_dir)[idx]
        imgA = cv2.imread(f'{self.image_dir}/{imgA_name}')
        imgA = cv2.resize(imgA, (160, 160))
        imgB_name = f'{imgA_name[:-4]}.png'
        imgB = cv2.imread(f'{self.mask_dir}/{imgB_name}', 0)
        imgB = cv2.resize(imgB, (160, 160))
        # imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 19)  # 19个 label ？ 还是18？？
        imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            imgA = self.transform(imgA)    

        return imgA, imgB

face = FaceDataLoader(image_dir='../data/CelebAMask-HQ/CelebA-HQ-img', mask_dir='../data/CelebAMask-HQ/mask', transform=transform)

train_size = int(0.9 * len(face))
test_size = len(face) - train_size
train_dataset, test_dataset = random_split(face, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
        