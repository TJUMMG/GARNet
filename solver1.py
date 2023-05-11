import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import transnet8 as network
from loss import IoU_loss, embaddingLoss
import numpy as np
import cv2
from dataset import get_loader
from os.path import join, exists
# import random
from utils import mkdir, write_doc, get_time, print_network
import torchvision.utils as vutils
from datetime import datetime

device = torch.device('cuda:0')

class Solver(object):
    def __init__(self):
        self.TransCosal = network.TransCosal().to(device)

    def train(self, roots1, init_epoch, end_epoch, learning_rate, batch_size, weight_decay, ckpt_root, doc_path, num_thread, pin, vgg_path=None):
        print_network(self.TransCosal)

        # bceLoss = torch.nn.BCEWithLogitsLoss()
        # Define Adam optimizer.
        optimizer = Adam(filter(lambda p: p.requires_grad, self.TransCosal.parameters()),
                         lr=learning_rate, 
                         weight_decay=weight_decay)

        # Load ".pth" to initialize model.
        if init_epoch == 0:
            # From pre-trained VGG16.
            self.TransCosal.apply(network.weights_init)
            self.TransCosal.vgg.vgg.load_state_dict(torch.load(vgg_path))
            # self.ICNet.vgg2.vgg.load_state_dict(torch.load(vgg_path))
        else:
            # From the existed checkpoint file.
            ckpt = torch.load(join(ckpt_root, 'Weights_{}.pth'.format(init_epoch)))
            self.TransCosal.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer']) 

        # Define training dataloader.
        train_dataloader = get_loader(roots=roots1,
                                      request=('img', 'gt', 'group_name'),
                                      shuffle=True,
                                      batch_size=batch_size,
                                      data_aug=True,
                                      num_thread=num_thread,
                                      pin=pin)
        
        train_dataloader1 = get_loader(roots=roots1,
                                request=('img', 'gt', 'group_name'),
                                shuffle=True,
                                batch_size=batch_size,
                                data_aug=True,
                                num_thread=num_thread,
                                pin=pin)

        total_step = len(train_dataloader)
        
        # Train. # 训练时返回三种数据量（原图，gt，sism）
        self.TransCosal.train()
        # embedding_features = None
        # group_name = None
        for epoch in range(init_epoch + 1, end_epoch):
            start_time = get_time()
            loss_sum = 0.0
            # total_step = len(train_dataloader)
            for i, data_batch in enumerate(zip(train_dataloader, train_dataloader1)): # 一个batch是同一组的图片
            # for i, data_batch in enumerate(train_dataloader): # 一个batch是同一组的图片
                self.TransCosal.zero_grad()

                # Obtain a batch of data.
                # img1, gt1 = data_batch[0]['img'], data_batch[0]['gt']
                img, gt, gn = data_batch[0]['img'], data_batch[0]['gt'], data_batch[0]['group_name']
                img1, gt1, gn1 = data_batch[1]['img'], data_batch[1]['gt'], data_batch[1]['group_name']
                # print(gn, gn1)
                # img, gt = data_batch['img'], data_batch['gt']
                img, gt = img.to(device), gt.to(device)
                img1, gt1 = img1.to(device), gt1.to(device)
                if gn == gn1:
                    label = 1
                else:
                    label = 0

                if len(img) == 1 or img.shape != img1.shape: # 如果只有一张图，跳过这次循环
                    # Skip this iteration when training batchsize is 1 due to Batch Normalization. 
                    continue
                
                # Forward.
                # print(img.shape, img1.shape)
                preds_list1, embedding_features1, preds_list2, embedding_features2 = self.TransCosal(
                        image_group1=img, image_group2 = img1, is_training=True)
                # features = F.interpolate(features, (224, 224), mode='bilinear', align_corners=True)


                loss1 = IoU_loss(preds_list1, gt)
                loss2 = embaddingLoss(embedding_features1, embedding_features1, 0)
                loss3 = IoU_loss(preds_list2, gt1)
                loss4 = embaddingLoss(embedding_features2, embedding_features2, 0)
                loss5 = embaddingLoss(embedding_features1, embedding_features2, label)
                

                # Backward.
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                loss.backward()
                optimizer.step()

                # 输出一些训练信息
                if i % 30 == 0 or i == total_step: 
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                            format(datetime.now(), epoch, (end_epoch - init_epoch), i, total_step, loss))
                tmp_path = './tmp_path_train/'
                if not exists(tmp_path):
                    mkdir(tmp_path)
                if i % 50 == 0:
                    # vutils.save_image(torch.sigmoid(preds_list.data), tmp_path +'iter%d-cosal.jpg' % i, normalize=True, padding = 0)
                    vutils.save_image(torch.sigmoid(preds_list1[len(preds_list1)-1].data), tmp_path +'iter%d-cosal.jpg' % i, normalize=True, padding = 0)
                    # vutils.save_image(torch.sigmoid(preds_list[0].data), tmp_path +'iter%d-attention.jpg' % i, normalize=True, padding = 0)
                    vutils.save_image(img.data, tmp_path +'iter%d-image.jpg' % i, padding = 0)
                    vutils.save_image(gt.data, tmp_path +'iter%d-cosal_gt.jpg' % i, padding = 0)
                    # vutils.save_image(torch.sigmoid(preds_list2[len(preds_list2)-1].data), tmp_path +'iter%d-sod_p.jpg' % i, padding = 0)
                    # vutils.save_image(img2.data, tmp_path +'iter%d-sodimage.jpg' % i, padding = 0)
                    # vutils.save_image(gt2.data, tmp_path +'iter%d-sodgt.jpg' % i, padding = 0)


                loss_sum = loss_sum + loss.detach().item() # pytorch的动态图机制，不加这个的话可能会导致显存爆炸, 阻断反向传播
            
            # Save the checkpoint file (".pth") after each epoch.
            mkdir(ckpt_root) # 创建模型保存地址，保存的是一个字典：优化器的参数、模型参数
            if epoch % 3 == 0:
                torch.save({'optimizer': optimizer.state_dict(),
                            'state_dict': self.TransCosal.state_dict()}, join(ckpt_root, 'Weights_{}.pth'.format(epoch)))
            
            # Compute average loss over the training dataset approximately.
            loss_mean = loss_sum / total_step
            end_time = get_time()
            print('**************************************')
            print("Mean loss of epoch {}: {}".format(epoch, loss_mean))
            print("Training time of epoch {}: {}".format(epoch, (end_time - start_time)))
            print('**************************************')

            # Record training information (".txt").
            content = 'CkptIndex={}:    TrainLoss={}    LR={}    Time={}\n'.format(epoch, loss_mean, learning_rate, end_time - start_time)
            write_doc(doc_path, content)
    
    def test(self, roots, ckpt_path, pred_root, num_thread, batch_size, original_size, pin):
        with torch.no_grad():            
            # Load the specified checkpoint file(".pth").
            state_dict = torch.load(ckpt_path)['state_dict']
            self.TransCosal.load_state_dict(state_dict)
            self.TransCosal.eval()
            
            # Get names of the test datasets.
            datasets = roots.keys()

            # time_test
            total_time = 0.0
            total_num = 0

            # Test ICNet on each dataset.
            for dataset in datasets:
                # Define test dataloader for the current test dataset.
                test_dataloader = get_loader(roots=roots[dataset], 
                                             request=('img', 'file_name', 'group_name', 'size'), 
                                             shuffle=False,
                                             data_aug=False, 
                                             num_thread=num_thread, 
                                             batch_size=batch_size, #batch_size为none 读取整组图片
                                             pin=pin)

                # Create a folder for the current test dataset for saving predictions.
                mkdir(pred_root)
                cur_dataset_pred_root = join(pred_root, dataset)
                mkdir(cur_dataset_pred_root)

                start_time = get_time()
                num = 0
                for data_batch in test_dataloader:
                    # Obtain a batch of data.
                    img = data_batch['img'].to(device)

                    # Forward.
                    preds = self.TransCosal(image_group1=img,  image_group2=None,
                                       is_training=False)
                    
                    
                    # Create a folder for the current batch according to its "group_name" for saving predictions.
                    group_name = data_batch['group_name'][0]
                    # print(data_batch['group_name'])
                    cur_group_pred_root = join(cur_dataset_pred_root, group_name)
                    mkdir(cur_group_pred_root)

                    # preds.shape: [N, 1, H, W]->[N, H, W, 1]
                    preds = preds.permute(0, 2, 3, 1).cpu().numpy()

                    # #热力图
                    # features = features.permute(0, 2, 3, 1).cpu().numpy()

                    # Make paths where predictions will be saved.
                    pred_paths = list(map(lambda file_name: join(cur_group_pred_root, file_name + '.png'), data_batch['file_name']))

                    num += len(pred_paths)
                    # For each prediction:
                    for i, pred_path in enumerate(pred_paths):
                        # Resize the prediction to the original size when "original_size == True".
                        H, W = data_batch['size'][0][i], data_batch['size'][1][i]
                        pred = cv2.resize(preds[i], (W, H)) if original_size else preds[i]
                        # headmap = cv2.resize(features[i], (W, H)) if original_size else features[i]

                        # Save the prediction.
                        cv2.imwrite(pred_path, np.array(pred * 255))
                        # cv2.imwrite(pred_path[:-4]+"_hm.png", np.array(headmap * 255))
                end_time = get_time()
                dataset_time = end_time - start_time
                print('*****************************************')
                print(dataset, num)
                print(dataset, dataset_time)
                print('*****************************************')
                total_num = total_num + num
                total_time = total_time + dataset_time
            print('*****************************************')
            print(total_num)
            print(total_time)
            print('*****************************************')
