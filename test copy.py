import os
from TransCosal.solver2 import Solver

# 测试时候的batchsize对性能的影响很大

"""
Test settings (used for "test.py"):

test_device:
    Index of the GPU used for test.

test_batch_size:
    Test batchsize.
  * When "test_batch_size == None", the dataloader takes the whole image group as a batch to
    perform the test (regardless of the size of the image group). If your GPU does not have enough memory,
    you are suggested to set "test_batch_size" with a small number (e.g. test_batch_size = 10).

pred_root:
    Folder path for saving predictions (co-saliency maps).

ckpt_path:
    Path of the checkpoint file (".pth") loaded for test.

original_size:
    When "original_size == True", the prediction (224*224) of ICNet will be resized to the original size.

test_roots:
    A dictionary including multiple sub-dictionary,
    each sub-dictionary contains the image and SISM folder paths of a specific test dataset.
    Format:
    test_roots = {
        name of dataset_1: {
            'img': image folder path of dataset_1,
            'sism': SISM folder path of dataset_1
        },
        name of dataset_2: {
            'img': image folder path of dataset_2,
            'sism': SISM folder path of dataset_2
        }
        .
        .
        .
    }
"""

test_device = '0'
test_batch_size = 8
# pred_root = './testresults/transnet8_all_SAM/prediction_70'
# ckpt_path = './ckpt/ckpt8_all_SAM/Weights_70.pth'
original_size = True
test_num_thread = 0
 
# An example to build "test_roots".
test_roots = dict()
datasets = ['CoSal2015', 'CoSOD3k']

for dataset in datasets:
    roots = {'img': '/media/HardDisk_new/wjx/datasets/{}/img/'.format(dataset)}
            #  'sism': /media/HardDisk_new/wjx/datasets/EGNet-SISMs-20201213T064828Z-001/EGNet-SISMs/{}/'.format(dataset)}
    test_roots[dataset] = roots
# ------------- end -------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = test_device
    for i in range(15):
        pred_root = './testresults/transnet8_one_4_ARM/prediction_{}'.format(48 + i * 3)
        ckpt_path = '/media/HardDisk_A/wjx/ckpt/ckpt_one_4_ARM//Weights_{}.pth'.format(48 + i * 3)
        solver = Solver()
        solver.test(roots=test_roots,
                    ckpt_path=ckpt_path,
                    pred_root=pred_root, 
                    num_thread=test_num_thread, 
                    batch_size=test_batch_size, 
                    original_size=original_size,
                    pin=False)
        
        print(f"weight {48 + i * 3} is done")
