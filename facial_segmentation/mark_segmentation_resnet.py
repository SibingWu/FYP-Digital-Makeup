#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
reference code:
https://github.com/zllrunning/face-parsing.PyTorch/blob/master/test.py
https://www.cxyzjd.com/article/u014453898/92080859
"""

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.vgg import VGG
from torchvision import transforms

import os
import sys  
sys.path.insert(0, os.getcwd())

# from FCN import FCNs, VGGNet


def get_mask(img_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     vgg_model = VGGNet(requires_grad=True, show_params=True)
#     fcn_model = FCNs(pretrained_net=vgg_model, n_class=19)  # 19 个 label
#     fcn_model = fcn_model.to(device)
    
    # fcn_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    fcn_model = fcn_resnet50(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 19
    filters_of_last_layer = fcn_model.classifier[4].in_channels
    filters_of_last_layer_aux = fcn_model.aux_classifier[4].in_channels
    fcn_model.classifier[4] = nn.Conv2d(filters_of_last_layer, num_classes, kernel_size=(1,1), stride=(1,1))
    fcn_model.aux_classifier[4] = nn.Conv2d(filters_of_last_layer_aux, num_classes, kernel_size=(1,1), stride=(1,1))
    fcn_model = fcn_model.to(device)
    
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    checkpoint = torch.load(checkpoint_path, map_location=device)
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
        output = fcn_model(img)['out']
        output = torch.sigmoid(output) # output.shape is torch.Size([4, 19, 160, 160])
        # print(type(output))
        # print(output.shape)

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
        # print(output_np.shape)   #(1, 19, 160, 160)
        output_np = np.argmax(output_np, axis=1)
        # print(output_np.shape)  #(1, 160, 160)

        return np.squeeze(output_np[0, ...])

        # plt.subplot(1, 2, 1)
        # #plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
        # #plt.subplot(1, 2, 2)
        # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
        # plt.pause(3)
        