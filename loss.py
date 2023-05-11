from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCELoss

"""
IoU_loss:
    Compute IoU loss between predictions and ground-truths for training [Equation 3].
经过测试，loss不能带权重, 不能是BCEloss
"""
def IoU_loss(preds_list, gt):
    preds = torch.cat(preds_list, dim=1)
    N, C, H, W = preds.shape
    # 作loss时gt会在通道反向自动补全 相当于多级监督
    min_tensor = torch.where(preds < gt, preds, gt)    # shape=[N, C, H, W]
    max_tensor = torch.where(preds > gt, preds, gt)    # shape=[N, C, H, W]
    min_sum = min_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    max_sum = max_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    loss = 1 - (min_sum / max_sum).mean() # 交并币损失
    return loss




class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)


class LossUsed(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super(LossUsed, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bceLoss = nn.BCEWithLogitsLoss()
        self.celLoss = CEL()   
    
    def forward(self, pred, gt):
        return self.alpha * self.bceLoss(pred, gt) + self.beta * self.celLoss(pred, gt)


# bce_loss = nn.BCELoss(size_average=True)
# ssim_loss = SSIM(window_size=11, size_average=True)
# def bce_ssim_loss(pred,target):
# 	bce_out = bce_loss(pred,target)
# 	ssim_out = 1 - ssim_loss(pred,target)
# 	iou_out = IoU_loss(pred,target)

# 	loss = bce_out + ssim_out + iou_out

# 	return loss

def embaddingLoss(feats1, feats2, label):
    feats1_norm = F.normalize(feats1, dim=1)
    feats2_norm = F.normalize(feats2, dim=1)
    cosine_similarity = torch.abs(torch.matmul(feats1_norm, feats2_norm.T))
    # cosine_similarity = torch.abs(torch.matmul(feats1, feats2.T))
    L = torch.mean(cosine_similarity)
    if label==0:
        return 1-L
    else:
        return L


class ContrastiveLoss(torch.nn.Module):
 
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
 
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
 
 
        return loss_contrastive


class Features_Loss(nn.Module):
    def __init__(self):
        super(Features_Loss,self).__init__()
    def forward(self, features, batch_size, person_num):
        loss_all = 0
        loss_temp = 0
        for id_index in range( batch_size // person_num ):
            features_temp_list = features[id_index*person_num:(id_index+1)*person_num+1]
            loss_temp = 0
            distance = torch.mm(features_temp_list,features_temp_list.t())
            distance = 1 -distance
            for i in range(person_num):
                for j in range( i + 1, person_num ):
                    loss_temp = loss_temp + distance[i][j]
            loss_temp = loss_temp / ( person_num * (person_num-1)/2 )
            loss_all = loss_all +  loss_temp
        loss_all = loss_all / ( batch_size // person_num )


# a = torch.randn([5, 128])
# b = torch.randn([5, 128])
# c = ContrastiveLoss()(a, b, 1)
# print(c)