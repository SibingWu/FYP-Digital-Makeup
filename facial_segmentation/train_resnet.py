#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
reference repo:
https://github.com/bat67/pytorch-FCN-easiest-demo/blob/master/train.py
"""

from datetime import datetime

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import visdom

from early_stopping import EarlyStopping
from face_dataloader import test_dataloader, train_dataloader
# from FCN import FCNs, VGGNet



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train(epoch_num=50, show_vgg_params=False):

    # vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    # fcn_model = FCNs(pretrained_net=vgg_model, n_class=19)  # 19 个 label
    # fcn_model = fcn_model.to(device)

    # download or load the model from disk
    fcn_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 19
    filters_of_last_layer = fcn_model.classifier[4].in_channels
    filters_of_last_layer_aux = fcn_model.aux_classifier[4].in_channels
    fcn_model.classifier[4] = nn.Conv2d(filters_of_last_layer, num_classes, kernel_size=(1,1), stride=(1,1))
    fcn_model.aux_classifier[4] = nn.Conv2d(filters_of_last_layer_aux, num_classes, kernel_size=(1,1), stride=(1,1))
    fcn_model = fcn_model.to(device)

    # # use weighted cross entropy loss for skewed dataset
    # # 对 dataset 中随机采样一张图
    # random_index = randint(0, len(os.listdir('/content/mask')))
    # random_img_path = '/content/mask/' + os.listdir('/content/mask')[random_index]
    # face_mask_example = cv2.imread(random_img_path, cv2.IMREAD_GRAYSCALE)
    # face_mask_example = cv2.resize(face_mask_example, (160, 160))
    # face_mask_example = face_mask_example.astype('uint8') 
    # df = pd.DataFrame(face_mask_example.flatten())
    # weights_dict = dict(df.value_counts())
    # weights = []
    # for i in range(num_classes):
    #     if (i,) in weights_dict.keys():
    #       weight = weights_dict[(i,)]
    #     else:
    #       weight = 0.1  # in case it is 0
    #     weights.append(weight)
    # print(weights)

    # weights = torch.tensor(weights, dtype=torch.float32)
    # # the more instance the less weight of a class
    # # normalize the weights proportionnally to the reverse of the initial weights
    # # If you want to reduce the number of false negatives then set β > 1
    # # similarly to decrease the number of false positives, set β < 1
    # weights = weights.sum() / weights
    # print(weights)
    # class_weights = torch.FloatTensor(weights).to(device)

    # [6345, 6266, 154, 150, 77, 63, 14, 12, 12, 18, 521, 42, 142, 181, 1556, 30, 104, 9913, 0.1]
    # tensor([4.0347e+00, 4.0856e+00, 1.6623e+02, 1.7067e+02, 3.3247e+02, 4.0635e+02,
    #     1.8286e+03, 2.1333e+03, 2.1333e+03, 1.4222e+03, 4.9136e+01, 6.0953e+02,
    #     1.8028e+02, 1.4144e+02, 1.6453e+01, 8.5334e+02, 2.4615e+02, 2.5825e+00,
    #     2.5600e+05])

    ## 不能像上面这么干，眼睛嘴唇这种小地方被除之后会非常dominant


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

    # the more instance the less weight of a class
    # normalize the weights proportionnally to the reverse of the initial weights
    # If you want to reduce the number of false negatives then set β > 1
    # similarly to decrease the number of false positives, set β < 1
    # 眼睛在10的时候，epoch 0就已经初见雏形了
    # 眉毛在20的时候，epoch 0就已经初见雏形，但是40会太粗
    weights = [0.5, 1.0, 30, 30, 20, 20, 10, 10, 10, 5, 3.0, 5, 10, 10, 1.0, 1.5, 1.0, 1.0, 1.3]
    weights = torch.tensor(weights, dtype=torch.float32)
    print(weights)
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True, path='archive/checkpoints_resnet/fcn_model_best.pt')

    # start timing
    prev_time = datetime.now()

    # # resume training
    checkpoint = torch.load('checkpoints_vgg/fcn_model_9.pt')
    fcn_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    for epoch in range(10, epoch_num):
        
        train_loss = 0
        fcn_model.train()

        # shuffle the dataset
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

        for index, (face, face_msk) in enumerate(train_dataloader):
            # face.shape is torch.Size([4, 3, 160, 160])
            # face_msk.shape is torch.Size([4, 19, 160, 160]) # 19 个 label
            # print('face_mask.shape:', face_msk.shape)

            face = face.to(device)
            # print(face_msk.shape)
            # print(face_msk)
            face_msk = face_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(face)['out']
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 19, 160, 160])
            loss = criterion(output, face_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 19, 160, 160)  
            output_np = np.argmax(output_np, axis=1)
            face_msk_np = face_msk.cpu().detach().numpy().copy() # face_msk_np.shape = (4, 19, 160, 160) 
            face_msk_np = np.argmax(face_msk_np, axis=1)
            # print(face_msk_np)

            if np.mod(index, 25) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epoch, index, len(train_dataloader), iter_loss))
                # vis.close()
                # vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                # vis.images(face_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                # vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

            if np.mod(index, 1000) == 0:
                plt.subplot(1, 2, 1) 
                plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                plt.subplot(1, 2, 2) 
                plt.imshow(np.squeeze(face_msk_np[0, ...]), 'gray')
                plt.pause(0.5)

        
        ##### validation #####
        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (face, face_msk) in enumerate(test_dataloader):

                face = face.to(device)
                face_msk = face_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(face)['out']
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 19, 160, 160])
                loss = criterion(output, face_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 19, 160, 160)  
                output_np = np.argmax(output_np, axis=1)
                face_msk_np = face_msk.cpu().detach().numpy().copy() # face_msk_np.shape = (4, 19, 160, 160) 
                face_msk_np = np.argmax(face_msk_np, axis=1)
        
                # if np.mod(index, 15) == 0:
                #     print(r'Testing... Open http://localhost:8097/ to see test result.')
                #     vis.close()
                #     vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                #     vis.images(face_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                #     vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

                if np.mod(index, 25) == 0:
                    plt.subplot(1, 2, 1) 
                    plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                    plt.subplot(1, 2, 2) 
                    plt.imshow(np.squeeze(face_msk_np[0, ...]), 'gray')
                    plt.pause(0.5)


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        

        if not os.path.exists('archive/checkpoints_resnet'):
            os.makedirs('archive/checkpoints_resnet')
        
        model_path = 'checkpoints_resnet/fcn_model_{}.pt'.format(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': fcn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)
        
        print(f'saving {model_path}')
            
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(test_loss, fcn_model)
        
        if early_stopping.early_stop:
            print("Early stopping......")
            break


if __name__ == "__main__":

    train(epoch_num=100, show_vgg_params=False)
