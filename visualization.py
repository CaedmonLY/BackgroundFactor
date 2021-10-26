


import os
import torch
import torch.nn as nn
import torch.utils.data as Data              # datalorder 的所在地
import torchvision
import numpy as np


from utlit import  saveModel,restoreModel
from getData import FileDataSet
from modle import myCNN

from modleToBeEval import modleFun





BATCH_SIZE = 1
# Data Loader for easy mini-batch return in training, the image batch shape will be (128, 3, 28, 28)



test_data =FileDataSet('./data', train=False)
test_loader    = Data.DataLoader(dataset=test_data,  batch_size=BATCH_SIZE, shuffle=True)




# 演示一张图　一张图　地测试。展示
import cv2

cnn=myCNN()
cnn.cuda()
restoreModel(cnn,'./model',info="boosted_")

from boost import randBoost
randboost=randBoost()

correctnum=0

for index, (testpic, testlable,mask) in enumerate(test_loader):
    testpic = testpic.cuda()
    testlable = testlable.cuda()
    mask = mask.cuda()

    testpic = randboost(testpic, mask)

    output = cnn(testpic)

    ans = torch.argmax(output, dim=1).item()

    flag="True"
    if(testlable[0][ans]!=1):
        flag="False"

    ans =int(ans)


    # 以上内容均在gpu上运行

    testpic =testpic.cpu()
    # chw->hwc
    pic =testpic.permute(0,2 ,3 ,1).contiguous().float().numpy()
    pic=(pic+1)*255/2
    pic = np.uint8(pic[0])  # 注意  pyotrch图片  是带batch的          使用opencv注意去掉batch

    # 放缩图片
    scale=3
    h,w,c=pic.shape
    pic=cv2.resize(pic,(int(w*scale),int(h*scale)))

    h, w, c = pic.shape
    img =torch.full((500 ,500 ,3) ,0.1).numpy()
    img = np.uint8(img)

    # 原始图帖到大画布上去
    dx,dy=10,150
    img[dx:h+dx ,dy:w+dy ,: ] =pic


    ans=str(ans)
    print("文件：",index," 预测值",ans)
    font = cv2.FONT_HERSHEY_DUPLEX                 # 定义字体


    img = cv2.putText(img, "img", (10, 30), font, 1, (255, 255, 255), 1)
    img = cv2.putText(img, "predict", (10, 150), font, 1, (255, 255, 255), 1)
    img = cv2.putText(img, "flag   ", (10, 200), font, 1, (255, 255, 255), 1)
    img = cv2.putText(img, ans, (150,150), font, 1, (255, 255, 255), 1)
    img = cv2.putText(img, flag, (150, 200), font, 1, (255, 255, 255), 1)



    cv2.imshow("s",img)
    cv2.waitKey(0)
    if(flag=="True"):
        #cv2.waitKey(0)
        correctnum=correctnum+1

print(correctnum)

cv2.destroyAllWindows()


"""
cv2 添加文字

import cv2
img = cv2.imread('caijian.jpg')
font = cv2.FONT_HERSHEY_SIMPLEX

imgzi = cv2.putText(img, '000', (50, 300), font, 1.2, (255, 255, 255), 2)
                   # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
"""