"""

完整的一个demo，从数据集构建，到训练，到可视化



毋文靖
2019/12/13
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
from modle import  myCNN

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH =10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 512
LR = 0.001             # learning rate

# 设置一个较大的初始值，没过10轮，折半



train_data=FileDataSet("./data",train=True,realBackground=False,boostObject="NULL")
test_data =FileDataSet('./data', train=False)

# Data Loader for easy mini-batch return in training, the image batch shape will be (128, 3, 28, 28)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE, shuffle=True)
test_loader    = Data.DataLoader(dataset=test_data,  batch_size=BATCH_SIZE, shuffle=False)

imgBoost=True






cnn = myCNN()
print(cnn)  # net architecture

# adam不太行。。。，结果不如sgd
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
#optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)    #  需要指明，是哪个模型的参数，以及学习率
                                                        # pytorch官网上有不用优化器的。 直接用for循环解决

loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_func=nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn=cnn.cuda()

from boost import randBoost
randboost=randBoost().cuda()
#restoreModel(cnn,'./model')
restoreModel(cnn,'./model',info="boosted_")
for epoch in range(EPOCH):
    for step, (b_x, b_y,mask) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

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

        if step % 50 == 0:
            totleImg=0
            correctNum=0
            correct=0

            for testpic,testlable,mask in test_loader :
                testpic=testpic.cuda()
                testlable=testlable.cuda()
                mask=mask.cuda()
                if imgBoost:
                    testpic=randboost(testpic, mask)
                    print(testpic.shape)
                output=cnn(testpic)





                """
                # 测试数据可视化
                testpic = testpic.cpu()
                # chw->hwc
                pic = testpic.permute(0, 2, 3, 1).contiguous().float().numpy()
                pic = (pic + 1) * 255 / 2
                pic = np.uint8(pic[0])  # 注意  pyotrch图片  是带batch的          使用opencv注意去掉batch
                # 放缩图片
                scale = 3
                h, w, c = pic.shape
                pic = cv2.resize(pic, (int(w * scale), int(h * scale)))
                h, w, c = pic.shape
                img = torch.full((500, 500, 3), 0.1).numpy()
                img = np.uint8(img)
                # 原始图帖到大画布上去
                dx, dy = 10, 150
                img[dx:h + dx, dy:w + dy, :] = pic
                cv2.imshow("s",img)
                cv2.waitKey(0)
                """










                # 这个写的太复杂了，直接判定最大值在那个下标就行

                # 最后一个batch可能达不到 BATCH_SIZE 的真实值    ,   因此要实时地从 dataset中获取
                # index0=torch.arange(BATCH_SIZE)
                trueBatch,c,h,w=testpic.shape

                index0 = torch.arange(trueBatch)
                index1=torch.argmax(output,dim=1)

                #print(output)
                #print(testlable)
                ans=torch.zeros([trueBatch,10]).cuda()                 # 这个错了，    怎么才能避免 漏写cuda()
                                                                        # 是不是应该吧 train，test都写成class，继承nn.Module。然后统一.cuda()

                """
                不可以按照0 填充
                lable 是 [0,0,1]
                ans     =[0,0,x]        那么，相等的总会有，至少2个
                
                除了考虑相等的，还要考虑不相等的那种怎么体现
                
                """

                ans=torch.full([trueBatch,10],-1).cuda()
                ''''
                print("debug index")
                print(output.shape)
                print(index0.shape)
                print(index1.shape)
                print("ok")
                '''

                ans[index0,index1]=1


                ans=(ans==testlable)
                correct=torch.sum(ans)         #tensor(9330, device='cuda:0')        这个得到的是tensor
                correctNum=correctNum+correct
                totleImg=totleImg+trueBatch

            correctNum=correctNum.item()             # 转换成小数

            accuracy=correctNum/totleImg
            loss = loss.cpu()
            print("正确的个数",correctNum)
            print("总图片",totleImg)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)







'''
        if step % 50 == 0:
            print("debug  dev")
            test_x.cuda()
            test_output= cnn(test_x)
            test_output=test_output.cpu()
            pred_y = torch.max(test_output, 1)[1].numpy()

            # 这里的演示代码用到 numpy，然而，numpy只能在cpu上跑， 模型本身的测试在gpu上跑。   跑完之后，在拷贝到cpu上计算剩下的东西
            test_y=test_y.cpu()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            loss=loss.cpu()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

'''





saveModel(cnn,'./model',info="boosted_")



# print 10 predictions from test data
'''
pridic=myCNN()
pridic.cuda()
restoreModel(pridic)

print("查看测试数据")
print(test_x[:10].shape)         # 整个一大批数据


test_output= pridic(test_x[:10])
pred_y = torch.max(test_output, 1)

print(pred_y, 'prediction number')

print(test_y[:10], 'real number')
'''

'''
todo 

torch.max() 函数的用法
返回值有2个？？？

'''













