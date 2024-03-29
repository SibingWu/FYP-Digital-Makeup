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
from FCN import FCNs, VGGNet


def train(epoch_num=50, show_vgg_params=False):

    # vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=19)  # 19 个 label
    fcn_model = fcn_model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)

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
            output = fcn_model(face)
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

            # if epoch % 15 == 0 and np.mod(index, 15) == 0:
            #     plt.subplot(1, 2, 1) 
            #     plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
            #     plt.subplot(1, 2, 2) 
            #     plt.imshow(np.squeeze(face_msk_np[0, ...]), 'gray')
            #     plt.pause(0.5)

        
        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (face, face_msk) in enumerate(test_dataloader):

                face = face.to(device)
                face_msk = face_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(face)
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

                # if epoch % 15 == 0 and np.mod(index, 25) == 0:
                #     plt.subplot(1, 2, 1) 
                #     plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
                #     plt.subplot(1, 2, 2) 
                #     plt.imshow(np.squeeze(face_msk_np[0, ...]), 'gray')
                #     plt.pause(0.5)


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        

        # if np.mod(epoch, 5) == 0:
            
        if not os.path.exists('archive/checkpoints_vgg'):
            os.makedirs('archive/checkpoints_vgg')
        
        model_path = 'checkpoints_vgg/fcn_model_{}.pt'.format(epoch)
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
