from re import S
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import transnet8 as network
from loss import IoU_loss, embaddingLoss, ContrastiveLoss
import numpy as np
import cv2
from dataset import get_loader
from dataSOD import get_loader as get_loader2
from os.path import join, exists
# import random
from utils import mkdir, write_doc, get_time, print_network
import torchvision.utils as vutils
from datetime import datetime
from sklearn.decomposition import PCA


class Solver(object):
    def __init__(self):
        self.TransCosal = network.TransCosal().cuda()

    def train(self, roots1, roots2, init_epoch, end_epoch, learning_rate, batch_size, weight_decay, ckpt_root, doc_path, num_thread, pin, vgg_path=None):
        print_network(self.TransCosal)
        cLoss = ContrastiveLoss()
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
        train_dataloader1 = get_loader(roots=roots1,
                                      request=('img', 'gt', 'group_name'),
                                      shuffle=True,
                                      batch_size=batch_size,
                                      data_aug=True,
                                      num_thread=num_thread,
                                      pin=pin)
        
        train_dataloader2 = get_loader2(image_root=roots2['img'],
                                      gt_root=roots2['gt'],
                                      trainsize=224,
                                      shuffle=True,
                                      batchsize=batch_size,
                                      pin_memory=pin)

        total_step = min(len(train_dataloader1), len(train_dataloader2))
        
        # Train. # 训练时返回三种数据量（原图，gt，sism）
        self.TransCosal.train()
        # embedding_features = None
        # group_name = None
        for epoch in range(init_epoch + 1, end_epoch):
            start_time = get_time()
            loss_sum = 0.0
            # total_step = len(train_dataloader)
            for i, data_batch in enumerate(zip(train_dataloader1, train_dataloader2)):

                self.TransCosal.zero_grad()
                img1, gt1 = data_batch[0]['img'], data_batch[0]['gt']
                img2, gt2 = data_batch[1]['img'], data_batch[1]['gt']
                img1, gt1 = img1.cuda(), gt1.cuda()
                img2, gt2 = img2.cuda(), gt2.cuda()

                if len(img1) == 1:
                    # Skip this iteration when training batchsize is 1 due to Batch Normalization. 
                    continue
                
                # Forward.
                preds_list1, embedding_features1, preds_list2, embedding_features2 = self.TransCosal(
                        image_group1=img1, image_group2 = img2, is_training=True)
                # preds_list1, preds_list2 = self.TransCosal(
                #         image_group1=img1, image_group2 = img2, is_training=True)

                loss1 = IoU_loss(preds_list1, gt1)
                # loss2 = embaddingLoss(embedding_features1, embedding_features1, 0)
                loss3 = IoU_loss(preds_list2, gt2)
                # loss4 = embaddingLoss(embedding_features2, embedding_features2, 1)
                # loss5 = embaddingLoss(embedding_features1, embedding_features2, 1)
                
                # Backward.
                # loss = loss1 + 0.05*loss2 + loss3 + 0.05*loss5
                loss = loss1 + loss3
                loss.backward()
                optimizer.step()

                # 输出一些训练信息
                if i % 30 == 0 or i == total_step: 
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss1: {:.4f}, Loss3: {:.4f}'.
                            format(datetime.now(), epoch, (end_epoch - init_epoch), i, total_step, loss, loss1, loss3))
                tmp_path = './tmp_path_train_two/'
                if not exists(tmp_path):
                    mkdir(tmp_path)
                if i % 50 == 0:
                    # vutils.save_image(torch.sigmoid(preds_list.data), tmp_path +'iter%d-cosal.jpg' % i, normalize=True, padding = 0)
                    vutils.save_image(torch.sigmoid(preds_list1[len(preds_list1)-1].data), tmp_path +'iter%d-cosal.jpg' % i, normalize=True, padding = 0)
                    # vutils.save_image(torch.sigmoid(preds_list[0].data), tmp_path +'iter%d-attention.jpg' % i, normalize=True, padding = 0)
                    vutils.save_image(img1.data, tmp_path +'iter%d-image.jpg' % i, padding = 0)
                    vutils.save_image(gt1.data, tmp_path +'iter%d-cosal_gt.jpg' % i, padding = 0)
                    # vutils.save_image(torch.sigmoid(preds_list2[len(preds_list2)-1].data), tmp_path +'iter%d-sod_p.jpg' % i, padding = 0)
                    # vutils.save_image(img2.data, tmp_path +'iter%d-sodimage.jpg' % i, padding = 0)
                    # vutils.save_image(gt2.data, tmp_path +'iter%d-sodgt.jpg' % i, padding = 0)


                loss_sum = loss_sum + loss.detach().item() # pytorch的动态图机制，不加这个的话可能会导致显存爆炸
            
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
            self.TransCosal.load_state_dict(state_dict, strict=False)
            self.TransCosal.eval()
            
            # Get names of the test datasets.
            datasets = roots.keys()

            # time_test
            total_time = 0.0
            total_num = 0
            pca = PCA(n_components=1)

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
                    img = data_batch['img'].cuda()

                    # Forward.
                    # preds, casmap5bf, casmap5af, casmap3bf, casmap3af, casmap1bf, casmap1af = self.TransCosal(image_group1=img,  image_group2=None,
                    #                    is_training=False)
                    preds = self.TransCosal(image_group1=img,  image_group2=None, is_training=False)
                    
                    # Create a folder for the current batch according to its "group_name" for saving predictions.
                    group_name = data_batch['group_name'][0]
                    # print(data_batch['group_name'])
                    cur_group_pred_root = join(cur_dataset_pred_root, group_name)
                    mkdir(cur_group_pred_root)

                    # preds.shape: [N, 1, H, W]->[N, H, W, 1]
                    preds = preds.permute(0, 2, 3, 1).cpu().numpy()

                    
                    # casmap5bf = casmap5bf.permute(0, 2, 3, 1).cpu().numpy()
                    # casmap3bf = casmap3bf.permute(0, 2, 3, 1).cpu().numpy()
                    # casmap1bf = casmap1bf.permute(0, 2, 3, 1).cpu().numpy()
                    # casmap5af = casmap5af.permute(0, 2, 3, 1).cpu().numpy()
                    # casmap3af = casmap3af.permute(0, 2, 3, 1).cpu().numpy()
                    # casmap1af = casmap1af.permute(0, 2, 3, 1).cpu().numpy()

                    # Make paths where predictions will be saved.
                    pred_paths = list(map(lambda file_name: join(cur_group_pred_root, file_name + '.png'), data_batch['file_name']))

                    num += len(pred_paths)
                    # For each prediction:
                    for i, pred_path in enumerate(pred_paths):
                        # Resize the prediction to the original size when "original_size == True".
                        H, W = (data_batch['size'][0][i]).item(), (data_batch['size'][1][i]).item()
                        # print(H, W)
                        pred = cv2.resize(preds[i], (W, H)) if original_size else preds[i]
                        # #热力图

                        # casmap5bf1 = casmap5bf[i]
                        # h, w, c = casmap5bf1.shape
                        # casmap5bf1 = casmap5bf1.reshape(-1, c)
                        # pca.fit(casmap5bf1)
                        # casmap5bf1 = pca.fit_transform(casmap5bf1)
                        # casmap5bf1 = casmap5bf1.reshape(h, w, 1)
                        # casmap5af1 = casmap5af[i]
                        # casmap5af1 = casmap5af1.reshape(-1, c)
                        # pca.fit(casmap5af1)
                        # casmap5af1 = pca.fit_transform(casmap5af1)
                        # casmap5af1 = casmap5af1.reshape(h, w, 1)

                        # casmap3bf1 = casmap3bf[i]
                        # h, w, c = casmap3bf1.shape
                        # casmap3bf1 = casmap3bf1.reshape(-1, c)
                        # pca.fit(casmap3bf1)
                        # casmap3bf1 = pca.fit_transform(casmap3bf1)
                        # casmap3bf1 = casmap3bf1.reshape(h, w, 1)
                        # casmap3af1 = casmap3af[i]
                        # casmap3af1 = casmap3af1.reshape(-1, c)
                        # pca.fit(casmap3af1)
                        # casmap3af1 = pca.fit_transform(casmap3af1)
                        # casmap3af1 = casmap3af1.reshape(h, w, 1)

                        # casmap1bf1 = casmap1bf[i]
                        # h, w, c = casmap1bf1.shape
                        # casmap1bf1 = casmap1bf1.reshape(-1, c)
                        # pca.fit(casmap1bf1)
                        # casmap1bf1 = pca.fit_transform(casmap1bf1)
                        # casmap1bf1 = casmap1bf1.reshape(h, w, 1)
                        # casmap1af1 = casmap1af[i]
                        # casmap1af1 = casmap1af1.reshape(-1, c)
                        # pca.fit(casmap1af1)
                        # casmap1af1 = pca.fit_transform(casmap1af1)
                        # casmap1af1 = casmap1af1.reshape(h, w, 1)

                        # casmap5bf1 = cv2.resize(casmap5bf1, (W, H)) if original_size else casmap5bf1 
                        # casmap3bf1 = cv2.resize(casmap3bf1, (W, H)) if original_size else casmap3bf1
                        # casmap1bf1 = cv2.resize(casmap1bf1, (W, H)) if original_size else casmap1bf1
                        # casmap5af1 = cv2.resize(casmap5af1, (W, H)) if original_size else casmap5af1
                        # casmap3af1 = cv2.resize(casmap3af1, (W, H)) if original_size else casmap3af1
                        # casmap1af1 = cv2.resize(casmap1af1, (W, H)) if original_size else casmap1af1

                        # # Save the prediction.
                        # casmap5bf1 = (casmap5bf1 - casmap5bf1.min()) / (casmap5bf1.max() - casmap5bf1.min())
                        # casmap5af1 = (casmap5af1 - casmap5af1.min()) / (casmap5af1.max() - casmap5af1.min())
                        # casmap3bf1 = (casmap3bf1 - casmap3bf1.min()) / (casmap3bf1.max() - casmap3bf1.min())
                        # casmap3af1 = (casmap3af1 - casmap3af1.min()) / (casmap3af1.max() - casmap3af1.min())
                        # casmap1bf1 = (casmap1bf1 - casmap1bf1.min()) / (casmap1bf1.max() - casmap1bf1.min())
                        # casmap1af1 = (casmap1af1 - casmap1af1.min()) / (casmap1af1.max() - casmap1af1.min())
                        
                        cv2.imwrite(pred_path, np.array(pred * 255))
                        # cv2.imwrite(pred_path[:-4]+"_hm5bf.png", np.array(casmap5bf1 * 255))
                        # cv2.imwrite(pred_path[:-4]+"_hm3bf.png", np.array(casmap3bf1 * 255))
                        # cv2.imwrite(pred_path[:-4]+"_hm1bf.png", np.array(casmap1bf1 * 255))
                        # cv2.imwrite(pred_path[:-4]+"_hm5af.png", np.array(casmap5af1 * 255))
                        # cv2.imwrite(pred_path[:-4]+"_hm3af.png", np.array(casmap3af1 * 255))
                        # cv2.imwrite(pred_path[:-4]+"_hm1af.png", np.array(casmap1af1 * 255))
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
