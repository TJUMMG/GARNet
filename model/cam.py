import torch

# 热力图计算
def camCaculate(features):
        cam = torch.mean(features, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-6)
        cam = torch.clamp(cam, 0, 1)
        return cam