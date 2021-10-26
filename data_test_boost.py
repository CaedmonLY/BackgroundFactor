"""

对手写数字进行背景无关增强

./data/train_boost/*.img
./data/train_boost.txt

./data/test_boost/*.img
./data/test_boost.txt


毋文靖
2019/12/19

"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

import os
import cv2
import numpy as np


# 下载、从文件中读取手写识别
# 这段代码来自莫凡python

DOWNLOAD_MNIST = False
# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)


'''
# 上面的方法可以读图片成tensor，可以正常训练，但是，不知道为什么，用cv2写图片的时候是一片黑。

# 没有问题，上面读出来的全部是[0,1]的数据


import torchvision.datasets.mnist as mnist
root=u"./"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )

'''

'''
可似乎数据集
for item in train_data:
    print(item[0].shape)
    print(item[1])

    pic=torch.tensor(item[0]).permute(1,2,0).contiguous().numpy()
    cv2.imshow("img",pic)
    cv2.waitKey(1)
cv2.destroyAllWindows()
'''





def SaveDataSet(dataset,datasetName="train"):
    """
    为了简化数据与lable的对应关系
    lable按照行存，一行存一个lable
    img的名字为lable的索引

    1994.img对应的lable为    train_lable[1994]

    要做到这一点，只需要img从0开始命名就行了

    """
    data_root_path='./data'
    if(os.path.exists(data_root_path)):
        pass
    else:
        os.mkdir(data_root_path)

    if (os.path.exists(data_root_path + '/' + datasetName)):
        pass
    else:
        os.mkdir(data_root_path + '/' + datasetName)
    imgDataRoot=data_root_path+'/'+datasetName+'/'
    lablefileName=data_root_path+'/'+datasetName+'.txt'

    lable=open(lablefileName,'w')

    for index,item in enumerate(dataset):
        #print(item[0].shape)     item[0]        [1,28,28] 型的tensor数据
        #print(item[1])           item[1]        int       型的标签

        pic=item[0]
        pic=pic.permute(1,2,0).contiguous()
        pic=pic.numpy()*255

        picName=imgDataRoot+str(index)+'.png'
        cv2.imwrite(picName,pic)
        lable.write(str(item[1])+'\n')

#        cv2.imshow("img", pic)
#        cv2.waitKey(1)
#    cv2.destroyAllWindows()


SaveDataSet(train_data,"train")
SaveDataSet(test_data,"test")




# 文件太多，sb win10直接卡死了
# 还是写个脚本方便
def removepic():
    file = os.listdir("./data/train")
    print(file[1:5])
    for i in file:
        os.remove("./data/train/" + i)