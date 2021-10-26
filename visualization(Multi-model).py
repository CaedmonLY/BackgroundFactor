


import os
import torch
import torch.nn as nn
import torch.utils.data as Data              # datalorder 的所在地
import torchvision
import numpy as np
import cv2
import time

from utlit import  saveModel,restoreModel
from getData import FileDataSet
from modle import myCNN

from modleToBeEval import modleFun







BATCH_SIZE = 1
# Data Loader for easy mini-batch return in training, the image batch shape will be (128, 3, 28, 28)



#dataSet1=FileDataSet('./data', train=False,realBackground=True)
#dataSet2=FileDataSet('./data', train=False,realBackground=False)
#test_loader    = Data.DataLoader(dataset=test_data,  batch_size=BATCH_SIZE, shuffle=True)

from boost import randBoost
randboost=randBoost()

# 演示一张图　一张图　地测试。展示

testSave="ans.txt"
savefile=open(testSave,"a")
savefile.write(time.ctime()+'\n')

def getTestEvalInfo(testType="NN",imgShow=False):
    """
    testType:
            NN: normal modle+ normal data
            BN: boost modle + normal data

            NR: normal modle+ real background data
            BR: boost modle + real background data

            NB: normal modle+ boost background data
            BB: boost  modle+ boost background data
    """



    modleInfo = "boosted"  # 背景增强训练的模型
    if(testType[0]=="N"):
        modleInfo="normal"
    DataSetLoader=[]
    dataSet = FileDataSet('./data', train=False, realBackground=False)
    DataSetLoader = Data.DataLoader(dataset=dataSet, batch_size=1, shuffle=True)
    if(testType[1]=="R"):
        dataSet=FileDataSet('./data', train=False,realBackground=True)
        DataSetLoader=Data.DataLoader(dataset=dataSet,  batch_size=1, shuffle=True)




    for funName in modleFun:
        modleName = funName
        cnnFunction = modleFun[funName]
        cnn = cnnFunction(pretrained=False, progress=False, num_classes=10)
        title="当前测试的模型： "+modleName+"，   当前测试问题的类型： "+testType+"  "
        cnn = cnn.cuda()
        cnn = cnn.eval()

        restoreModel(cnn,'./model',info=modleInfo+"_"+modleName)
        #restoreModel(cnn, './model', info="4_0boosted_resnet18")

        correctnum = 0

        for index, (testpic, testlable, mask) in enumerate(DataSetLoader):
            testpic = testpic.cuda()
            testlable = testlable.cuda()
            mask = mask.cuda()

            if (testType == "BB" or testType == "NB"):
                testpic = randboost(testpic, mask)

            output = cnn(testpic)

            ans = torch.argmax(output, dim=1).item()

            flag = "True"
            if (testlable[0][ans] != 1):
                flag = "False"

            ans = int(ans)

            # 以上内容均在gpu上运行

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

            ans = str(ans)
            # print("文件：", index, " 预测值", ans)
            font = cv2.FONT_HERSHEY_DUPLEX  # 定义字体

            img = cv2.putText(img, "img", (10, 30), font, 1, (255, 255, 255), 1)
            img = cv2.putText(img, "predict", (10, 150), font, 1, (255, 255, 255), 1)
            img = cv2.putText(img, "flag   ", (10, 200), font, 1, (255, 255, 255), 1)
            img = cv2.putText(img, ans, (150, 150), font, 1, (255, 255, 255), 1)
            img = cv2.putText(img, flag, (150, 200), font, 1, (255, 255, 255), 1)

            if(imgShow):
                "可视化显示，每种情况显示一个demo"
                cv2.imshow("s",img)
                cv2.waitKey(0)
                return 0

            if (flag == "True"):
                # cv2.waitKey(0)
                correctnum = correctnum + 1

        title=title+"正确率： "+str(correctnum/100)
        print(title)
        savefile.write(title+'\n')

        cv2.destroyAllWindows()

        """
        cv2 添加文字

        import cv2
        img = cv2.imread('caijian.jpg')
        font = cv2.FONT_HERSHEY_SIMPLEX

        imgzi = cv2.putText(img, '000', (50, 300), font, 1.2, (255, 255, 255), 2)
                           # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
        """

# 实验测试类型
testType=[
    "NN",
    "NR",
    "NB",
    "BN",
    "BR",
    "BB"
]

demo=[
    "BN",
    "BR",
    "BB"
]

for t in demo:

    #getTestEvalInfo(t,imgShow=False)

    pass


for auto in testType:
    getTestEvalInfo(auto,imgShow=False)

    pass


savefile.close()





