import os
import torch
import time
import torch.nn as nn

"""
mkdir:
    Create a folder if "path" does not exist.
"""
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

"""
write_doc:
    Write "content" into the file(".txt") in "path".
"""
def write_doc(path, content):
    with open(path, 'a') as file:
        file.write(content)

"""
get_time:
    Obtain the current time.
"""
# 正确的测试时间的方法
def get_time():
    torch.cuda.synchronize()
    return time.time()


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()  # 参数量
    print("The number of parameters: {}".format(num_params))
