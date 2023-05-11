from numpy.core.fromnumeric import size
import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from os.path import join
from .transformer import TransformerCBA, TransformerCLA
from .cam import camCaculate
np.set_printoptions(suppress=True, threshold=1e5) 

#交替训练, 无loss
"""
resize:
    Resize tensor (shape=[N, C, H, W]) to the target size (default: 224*224).
"""
def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)


"""
weights_init:
    Weights initialization.
"""
def weights_init(module):
    if isinstance(module, nn.Conv2d):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)  


""""
VGG16:
    VGG16 backbone.
""" 
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        in_channel = 3
        vgg_out_channels = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
        for out_channel in vgg_out_channels:
            if out_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = out_channel
        self.vgg = nn.ModuleList(layers) # 对于连个列表来说，+ 表示连接 # 每个conv后面有一个relu
        self.table = {'conv1_1': 0, 'conv1_2': 2, 'conv1_2_mp': 4,
                      'conv2_1': 5, 'conv2_2': 7, 'conv2_2_mp': 9,
                      'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_3_mp': 16,
                      'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 'conv4_3_mp': 23,
                      'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28, 'conv5_3_mp': 30, 'final': 31}

    def forward(self, feats, start_layer_name, end_layer_name):
        start_idx = self.table[start_layer_name]
        end_idx = self.table[end_layer_name]
        for idx in range(start_idx, end_idx):
            feats = self.vgg[idx](feats) # 将特征送进vgg的start_layer层到end_layer层，输出最后一层的特征。
        return feats

"""
Prediction:
    Compress the channel of input features to 1, then predict maps with sigmoid function.
"""
class Prediction(nn.Module):
    def __init__(self, in_channel):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_channel, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        pred = self.pred(feats)
        return pred


"""
Res:
    Two convolutional layers with residual structure.
"""
class Res(nn.Module):
    def __init__(self, in_channel, size):
        super(Res, self).__init__()
        # self.conv0 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        #                           nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(2*in_channel, in_channel, 3, 1, 1), 
                                  nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.transformerCLA = TransformerCLA(d_model=256, nhead=4, num_decoder_layers=2, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=True, return_intermediate_dec=False)
        self.size = size

    def forward(self, feats1, feats2, feats3):
        feats1_t = F.interpolate(feats1, scale_factor=self.size)
        feats1_t = self.transformerCLA(feats1_t, feats3, None, None, None)
        feats1 = F.relu(feats1 + F.interpolate(feats1_t, scale_factor=1/self.size))
        feats2 = self.conv1(F.interpolate(feats2, scale_factor=2))
        feats = self.conv2(torch.cat([feats1, feats2], dim=1)) # 两个卷积层的残差结构
        return feats

"""
Refinement:
    U-net like decoder block that fuses co-saliency features and low-level features for upsampling. 
"""
class Decoder_Block(nn.Module):
    def __init__(self, in_channel):
        super(Decoder_Block, self).__init__()
        self.cmprs = nn.Conv2d(in_channel, 32, 1)

        self.conv = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.merge_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pred = Prediction(32)

    def forward(self, low_level_feats, old_feats, old_cosal_map):
        _, _, H, W = low_level_feats.shape
        old_feats = resize(old_feats, [H, W])
        old_cosal_map = resize(old_cosal_map, [H, W])
        # Predict co-saliency maps with the size of H*W.
        cmprs = self.cmprs(low_level_feats)
        cmprs = F.relu(cmprs + self.conv(old_cosal_map*cmprs))
        new_feats = self.merge_conv(torch.cat([cmprs, old_feats], dim=1))
        new_cosal_map = self.pred(new_feats)
        return new_feats, new_cosal_map


class Flaten_Module(nn.Module):
    def __init__(self, size, in_channel, factor):
        super(Flaten_Module, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((size, size))
        self.fn = nn.Sequential(
            nn.Linear(in_channel*size*size, factor),
            nn.ReLU(inplace=True),
            nn.Linear(factor, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        temp = self.pool(feats)
        temp = temp.flatten(1)
        return self.fn(temp)

"""
ICNet:
    The entire ICNet.
    Given a group of images and corresponding SISMs, ICNet outputs a group of co-saliency maps (predictions) at once.
"""
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4_cmprs = nn.Conv2d(512, 256, 1)
        self.fn = Flaten_Module(3, 256, 512)
        self.merge_co_45 = Res(256, 0.5)
        self.merge_co_34 = Res(256, 0.5)
        self.conv_out = nn.Conv2d(256, 32, 1)

        self.get_pred_3 = Prediction(32)
        
        self.refine_2 = Decoder_Block(128)
        self.refine_1 = Decoder_Block(64)

    def forward(self, conv1_2, conv2_2, conv3_3, conv4_3, conv5_cmprs, is_training):
        embedding_features = self.fn(conv5_cmprs)

        # print(conv4_3.shape)  # torch.Size([8, 512, 28, 28])
        conv4_cmprs = self.conv4_cmprs(conv4_3) 
        #conv4_cmprs, conv5_cmprs    torch.Size([8, 256, 28, 28]) torch.Size([8, 256, 14, 14])
        feat_45 = self.merge_co_45(conv4_cmprs, conv5_cmprs, conv5_cmprs)  
        # conv3_3, feat_45    torch.Size([8, 256, 56, 56]) torch.Size([8, 256, 28, 28])
        feat_34 = self.merge_co_34(conv3_3, feat_45, feat_45)  
        feat_34 = self.conv_out(feat_34)

        cosal_map_3 = self.get_pred_3(feat_34)  
        # print(cosal_map_3.shape)        #  torch.Size([2, 1, 56, 56])                           

        # print('3', conv2_2.shape, feat_34.shape)   torch.Size([8, 128, 112, 112])  torch.Size([8, 32, 56, 56]) 
        feat_23, cosal_map_2 = self.refine_2(conv2_2, feat_34, cosal_map_3) # 输出第二层的特征和特征预测
        # print(cosal_map_2.shape) # torch.Size([2, 1, 112, 112])
        # print('4', conv1_2.shape, feat_23.shape)   torch.Size([8, 64, 224, 224])   torch.Size([8, 32, 112, 112]) 
        _, cosal_map_1 = self.refine_1(conv1_2, feat_23, cosal_map_2)      # shape=[N, 1, 224, 224]最终的输出
        # print(_.shape) torch.Size([8, 32, 224, 224])

        # Return predicted co-saliency maps.
        if is_training:
            preds_list = [resize(cosal_map_3), resize(cosal_map_2), cosal_map_1]
            return preds_list, embedding_features
        else:
            preds = cosal_map_1
            return preds


class TransCosal(nn.Module):
    def __init__(self):
        super(TransCosal, self).__init__()
        self.vgg = VGG16()
        self.transformer = TransformerCBA(d_model=256, nhead=4, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=True)
        
        self.conv5_cmprs = nn.Conv2d(512, 256, 1)
        self.decoder = Decoder()

    def forward(self, image_group1, image_group2, is_training = True):
        
        conv1_2 = self.vgg(image_group1, 'conv1_1', 'conv1_2_mp') 
        conv2_2 = self.vgg(conv1_2, 'conv1_2_mp', 'conv2_2_mp')  
        conv3_3 = self.vgg(conv2_2, 'conv2_2_mp', 'conv3_3_mp')  
        conv4_3 = self.vgg(conv3_3, 'conv3_3_mp', 'conv4_3_mp')  
        conv5_3 = self.vgg(conv4_3, 'conv4_3_mp', 'conv5_3_mp')  
        conv5_cmprs = self.conv5_cmprs(conv5_3)
        conv5_cmprs = self.transformer(conv5_cmprs, None, None)  

        if is_training:
            conv1_2_2 = self.vgg(image_group2, 'conv1_1', 'conv1_2_mp') 
            conv2_2_2 = self.vgg(conv1_2_2, 'conv1_2_mp', 'conv2_2_mp')  
            conv3_3_2 = self.vgg(conv2_2_2, 'conv2_2_mp', 'conv3_3_mp')  
            conv4_3_2 = self.vgg(conv3_3_2, 'conv3_3_mp', 'conv4_3_mp')  
            conv5_3_2 = self.vgg(conv4_3_2, 'conv4_3_mp', 'conv5_3_mp')  
            conv5_cmprs_2 = self.conv5_cmprs(conv5_3_2) 
            preds_list1, embedding_features1 = self.decoder(conv1_2, conv2_2, conv3_3, conv4_3, conv5_cmprs, is_training)
            preds_list2, embedding_features2 = self.decoder(conv1_2_2, conv2_2_2, conv3_3_2, conv4_3_2, conv5_cmprs_2, is_training)
            return preds_list1, embedding_features1, preds_list2, embedding_features2
        else:
            return self.decoder(conv1_2, conv2_2, conv3_3, conv4_3, conv5_cmprs, is_training)


