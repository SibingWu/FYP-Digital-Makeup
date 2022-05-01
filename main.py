# -*- coding: utf-8 -*-
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from facial_segmentation.mark_segmentation_vgg import get_mask
# from facial_segmentation.mark_segmentation_resnet import get_mask
from colour_transfer.archive.colour_transfer import colour_transfer_by_colour_embedding


facial_feature_map = {
    'background': 0,
    'skin': 1,
    'l_brow': 2,
    'r_brow': 3,
    'l_eye': 4,
    'r_eye': 5,
    'eye_g': 6,
    'l_ear': 7,
    'r_ear': 8,
    'ear_r': 9,
    'nose': 10,
    'mouth': 11,
    'u_lip': 12,
    'l_lip': 13,
    'neck': 14,
    'neck_l': 15,
    'cloth': 16,
    'hair': 17,
    'hat': 18,
}


def facial_segmentation(img_original_path, img_target_path, facial_feature):
    """
    Gets masked segmentation
    :return:
    """
    # TODO: 研究一下这里为什么要 resize 成 160
    img_original = plt.imread(img_original_path)
    img_original = cv2.resize(img_original, (160, 160))
    # img_original_height, img_original_width = img_original.shape[0], img_original.shape[1]
    img_target = plt.imread(img_target_path)
    img_target = cv2.resize(img_target, (160, 160))
    # img_target = cv2.resize(img_target, (img_original_height, img_original_width))

    checkpoint_path = 'facial_segmentation/checkpoints_vgg_weightedCE/fcn_model_45.pt'
    # checkpoint_path = 'facial_segmentation/checkpoints_resnet_weightedCE/fcn_model_21.pt'
    img_original_mask = get_mask(img_path=img_original_path, checkpoint_path=checkpoint_path)
    img_target_mask = get_mask(img_path=img_target_path, checkpoint_path=checkpoint_path)

    # keep only targeted pixel
    facial_feature_code = facial_feature_map[facial_feature]
    img_source = img_original.copy()
    img_source[..., :][img_original_mask != facial_feature_code] = 255
    img_target[..., :][img_target_mask != facial_feature_code] = 255

    return img_original, img_source, img_target


def colour_transfer(img_original, img_source, img_target):
    """
    Apply colour transfer
    Reference: https://github.com/zhaohengyuan1/Color2Embed
    """
    # apply colour transfer
    img_transferred = colour_transfer_by_colour_embedding(img_source, img_target)

    # 把转换了的 facial feature 粘回去
    img_result = img_original.copy()
    img_result = img_result.reshape(-1, img_result.shape[-1])
    idx = np.argwhere(img_source.reshape(-1, img_source.shape[-1])[..., :] != 255)
    img_result[idx.flatten()] = img_transferred.reshape(-1, img_transferred.shape[-1])[idx.flatten()]
    img_result = img_result.reshape(img_source.shape)

    # visualize the result
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_original)
    ax[0].set_title('source')
    ax[1].imshow(img_target)
    ax[1].set_title('target')
    ax[2].imshow(img_result)
    ax[2].set_title('result')

    fig.savefig('results/source_target_result.png')
    plt.clf()

    return img_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_image_path', required=True)
    parser.add_argument('--target_image_path', required=True)
    parser.add_argument('--facial_feature', required=True)
    parser.add_argument('--result_path', required=False)
    args = parser.parse_args()

    # facial segmentation
    print('Segmenting Facial Feature...')
    img_original, img_source, img_target = \
        facial_segmentation(args.source_image_path, args.target_image_path, args.facial_feature)

    # colour transfer
    print('Transferring Colour...')
    img_result = colour_transfer(img_original, img_source, img_target)

    plt.imshow(img_result)
    if args.result_path:
        plt.savefig(args.result_path)
    else:
        plt.savefig('results/result.png')

    print('Result Image Saved!')


if __name__ == '__main__':
    main()
