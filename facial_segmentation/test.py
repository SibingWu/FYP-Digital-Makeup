#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
reference code:
https://github.com/zllrunning/face-parsing.PyTorch/blob/master/test.py
https://www.cxyzjd.com/article/u014453898/92080859
"""

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

from FCN import FCNs, VGGNet

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(img_path, checkpoint_path, result_dir):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vgg_model = VGGNet(requires_grad=True, show_params=True)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=19)  # 19 个 label
    fcn_model = fcn_model.to(device)
    
    
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    checkpoint = torch.load(checkpoint_path)
    fcn_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # model = torch.load(checkpoint_path)
    fcn_model.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        # for image_path in os.listdir(dspth):
        img = cv2.imread(img_path)
        image = cv2.resize(img, (160, 160))
        img = to_tensor(image)
        img = img.to(device)
        img = torch.unsqueeze(img, 0)

        # out = fcn_model(img)[0]
        output = fcn_model(img)
        output = torch.sigmoid(output) # output.shape is torch.Size([4, 19, 160, 160])
        print(type(output))
        print(output.shape)

        # parsing = torch.squeeze(output)
        # print(parsing.shape)
        # print(parsing)
        # print(parsing[0].shape)
        # print(parsing[0].argmax(dim=0))
        # parsing = parsing.cpu().detach().numpy().argmax(0)
        # print(parsing)
        # # print(parsing.shape)
        # # print(np.unique(parsing))

        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(result_dir, 'result.jpg'))

        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 19, 160, 160)
        print(output_np.shape)   #(1, 19, 160, 160)
        output_np = np.argmax(output_np, axis=1)
        print(output_np.shape)  #(1, 160, 160)

        plt.subplot(1, 2, 1)
        #plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
        #plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
        plt.pause(3)

        vis_parsing_maps(image, np.squeeze(output_np[0, ...]), stride=1, save_im=True, save_path=osp.join(result_dir, 'result.jpg'))


if __name__ == "__main__":
    evaluate(img_path='../data/CelebAMask-HQ/CelebA-HQ-img/0.jpg', checkpoint_path='archive/checkpoints_vgg/fcn_model_21.pt', result_path='../data/CelebAMask-HQ/inference_result')
    