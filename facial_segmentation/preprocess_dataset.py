#!/usr/bin/python
# -*- encoding: utf-8 -*-

# reference: https://github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py

import os.path as osp
import os
import cv2
import numpy as np
from transform import *
from PIL import Image

face_data = '../data/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '../data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
output_mask_path = '../data/CelebAMask-HQ/mask'

if not os.path.exists(output_mask_path):
    os.makedirs(output_mask_path)

counter = 0
total = 0

for i in range(15):  # 15 个 sub dir

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    '''
    0: background
    1: skin
    2: l_brow
    3: r_brow
    4: l_eye
    5: r_eye
    6: eye_g
    7: l_ear
    8: r_ear
    9: ear_r
    10: nose
    11: mouth
    12: u_lip
    13: l_lip
    14: neck
    15: neck_l
    16: cloth
    17: hair
    18: hat
    '''

    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))

        # l 为 index，用作最后 segmentation 的 label
        for l, att in enumerate(atts, 1):  # 从 index=1 开始, skin 的 index 认为是1
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))

                mask[sep_mask == 225] = l  # 染色标 label
        cv2.imwrite('{}/{}.png'.format(output_mask_path, j), mask)
        print(j)

print(counter, total)
