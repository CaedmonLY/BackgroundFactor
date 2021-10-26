"""

完整的一个demo，从数据集构建，到训练，到可视化



毋文靖
2019/12/13


和minist_副本的不同点是周期测试的那部分，删除了这部分代码，使用eval.py来替代它

"""

import os

# third-party library
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data as Data              # datalorder 的所在地
import torchvision
import matplotlib.pyplot as plt

from utlit import  saveModel,restoreModel
from getData import FileDataSet


from modleToBeEval import modleFun

import eval

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH =4               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 256
#LR = 0.001             # learning rate
LR = 0.001                # 对于mnasnet系列，学习率必须设置的大一点,
# 设置一个较大的初始值，没过10轮，折半

imgBoost=False
modleStoreInfo="normal"     # 无增强
if(imgBoost):
    modleStoreInfo="boosted"

evalRate=int(50*512/BATCH_SIZE)             # eval 频率


train_data=FileDataSet("./data",train=True,realBackground=False,boostObject="NULL")
test_data =FileDataSet('./data', train=False)

# Data Loader for easy mini-batch return in training, the image batch shape will be (128, 3, 28, 28)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE, shuffle=True)
test_loader    = Data.DataLoader(dataset=test_data,  batch_size=BATCH_SIZE, shuffle=False)


for funName in modleFun:

    modleName=funName
    cnnFunction=modleFun[funName]
    cnn = cnnFunction(pretrained=False, progress=False,num_classes=10)

    print(modleName)
    #continue

    # print(cnn)  # net architecture

    # adam不太行。。。，结果不如sgd                        todo  adam在保存再训练时有个隐藏的问题， 断点训练时，不能只加载模型的权重，
    #                                                        adam也有一个状态，这个也是需要保存的，断点训练时需要加载这个状态，
    #                                                        否则，再次训练时，由于自适应的关系，此时，真实的学习率仍然很大,
    #                                                        sgd不存在这个问题
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    #optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)    #  需要指明，是哪个模型的参数，以及学习率
                                                            # pytorch官网上有不用优化器的。 直接用for循环解决

    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    loss_func=nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn=cnn.cuda()

    from boost import randBoost
    randboost=randBoost().cuda()
    # restoreModel(cnn,'./model')           这个似乎是不管这么写都行，   python都是引用，函数内部对传入的对象做了变动，不需要以返回值的形式也行。
    #cnn=restoreModel(cnn,'./model',info="boosted_"+modleName)

    # 用于测试，保存最佳的模型
    bestans=0
    # 保存训练过程，便于后期分析
    xunlianguocheng=open("./log/trainlog.txt","w")
    import time
    # 格式化成2020-12-01@21:44:07形式
    now=time.strftime("%Y-%m-%d@%H:%M:%S", time.localtime())
    xunlianguocheng.write(now)
    xunlianguocheng.write("\n")

    for epoch in range(EPOCH):
        for step, (b_x, b_y,mask) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            cnn=cnn.train()
            b_x=b_x.cuda()
            b_y=b_y.cuda()
            mask=mask.cuda()

            '''
            # 用小批次模拟大批次得这么写
            b_x = randboost(b_x, mask)
            output = cnn(b_x)  # cnn output
            optimizer.zero_grad()  # clear gradients for this training step
            loss =  loss_func(output, b_y)  # cross entropy loss
            '''
            for i in range(100):
                if imgBoost:
                    b_x=randboost(b_x,mask)
                output = cnn(b_x)              # cnn output
                '''
                print("debug")
                print(output.shape)
                print(b_y.shape)
                '''
                optimizer.zero_grad()  # clear gradients for this training step
                loss = loss_func(output, b_y)   # cross entropy loss

                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                if(i%20==0):
                    print("batch ",step," loss",loss.cpu().item())

            if step % evalRate == 0:
                accuracy=eval.myEval(cnn,boost=imgBoost)
                cnn=cnn.train()
                print("总图片",accuracy)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().item(), '| test accuracy: %.4f' % accuracy)


                log="epoch {:d} loss {:.6f} test_accuracy {:.5f}\n".format(
                    epoch,loss.cpu().data.numpy(),accuracy
                )
                print(log)
                xunlianguocheng.write(log)

                if(accuracy>bestans):
                    bestans=accuracy
                    extname=str(epoch)+"_"+str(step)
                    #saveModel(cnn, './model', info=extname+modleStoreInfo+"_"+modleName)           添加了很多实验，不适合分析整个过程了，因此，直接保存最优

                    saveModel(cnn, './model', info=modleStoreInfo + "_" + modleName)


