"""
在“常规数据集合”上写dataset类

dataset 实际上是一个迭代器， 和dataloader配合高效利用内存

毋文靖
2019/12/12
"""

import torch
from torch.utils.data.dataset import Dataset
import numpy
import cv2

class FileDataSet(Dataset):

    # 这个init 接口怎么写完全取决于下面两个函数实现的时候需要什么数据，然后取初始化
    def __init__(self, dataRoot,realBackground=False,boostObject="background",train=True):
        # boostObject:  增强数据的类型，      NULL:无操作       object:增强检测对象
        # realBackground 是否使用真是存在的数据去替换背景
        # background:增强背景           2020/11/30.     这个没啥用了， 不知道怎么合理地加到cuda上运行。   转为直接在训练、测试的代码里加一行代码
        imgDir=""
        lable=""
        if(train):
            imgDir=dataRoot+"/train/"
            lable=dataRoot+"/train.txt"
        else:
            imgDir=dataRoot+"/test/"
            lable = dataRoot + "/test.txt"             #   第一次写，竟然写成了train.txt     赋值粘贴太容易出错了！！！！！！！！
                                                    #  不要偷懒

        self.lable=open(lable)
        self.lable=[int(item[:-1]) for item in self.lable]
        self.imgPath=imgDir

        self.boostObject=boostObject

        # 这里实际每啥用了，高斯增强那块直接放训练代码里了
        if(boostObject!="object" and boostObject!="background"):
            if(train):
                print("train set 没有使用随机数据增强")
            else:
                print("test set 没有使用随机数据增强")
            print("""
            boostObject 可选的增强方式有
                                        "object"
                                        "background"
            
            """)


        # 编码成一个10位的“二进制”
        self.trueLable=torch.zeros(10)


        # beg           这部分时为了验证在现实中图片的识别准确率的
        self.realBackground=realBackground
        mixpic=open("./tmp/test.txt",'r')
        self.filelist=[]
        for item in mixpic:
            item=item.replace("\n","")
            self.filelist.append(item)







    def __len__(self) -> int:
        return len(self.lable)

# 关键是写出这个函数，函数接口不能变，   根据索引i,  函数返回一个训练数据

    def __getitem__(self, i: int):
        # hwc
        #imgsize=(80,80)
        imgsize = (28, 28)
        pic=cv2.imread(self.imgPath+str(i)+'.png')
        # 为了适应vgg等的训练，临时增加了一个放缩图像的。
        #pic=cv2.resize(pic,imgsize)



        # 是否需要用真实存在的背景替换原来的随机噪声背景
        if self.realBackground:
            mixpic=cv2.imread("./tmp/9/"+self.filelist[i])
            mixpic=cv2.resize(mixpic,imgsize)

            tmp=torch.from_numpy(pic)
            index=torch.where(tmp>0)
            mixpic[index]=255
            pic=mixpic


        mask=self.getMask(pic)
        # chw
        tensorPic=torch.from_numpy(pic).permute(2,0,1).float().contiguous()
        tensormask=torch.from_numpy(mask).float()
        lable=int(self.lable[i])
        trueLable=torch.zeros(10)
        trueLable[lable]=1
        #  数据预处理
        #  演示，数据预处理的位置

        tensorPic=self.trans(tensorPic)

        return [tensorPic,trueLable,tensormask]


    # 数据预处理
    def trans(self,tensor):
        data=tensor/255*2-1
        return data

    def getMask(self,CV_img):

        gray=cv2.cvtColor(CV_img,cv2.COLOR_BGR2GRAY)
        tensor_gray=torch.from_numpy(gray).float()


        background=torch.where(tensor_gray<10)         # 选出0
        onject    =torch.where(tensor_gray>10)

        tensor_gray[background]=0
        tensor_gray[onject]=1

        return tensor_gray.numpy()


    #    已弃用，这部分代码直接在detaloader里的batch直接做
    def imgboost(self,img_tensor,mask_tensor,boostObject):
        """
        batchimg :   [3,h,w]         float
        btachmask:   [h,w]               float

        boostObject:  增强数据的类型，          object:增强检测对象       background:增强背景

        return  :    [3,h,w]
        """
        img=img_tensor.float()
        mask=mask_tensor.float()
        mask = torch.stack((mask, mask, mask), 0)      # 也构造成三通道，索引的时候直接和img匹配

        shape=img.shape
        boostTensor=torch.randn(shape)               # 标准正态分布中，大于3.9的部分，概率基本为0了。可以近似认为抽出来的数大致在[-3.9,3.9]
        boostTensor=boostTensor/3.9

        if(boostObject=="background"):
            index=torch.where(mask<0.5)         #  float数，还是确定一个范围比较保险，选出 ==0 的index
        elif(boostObject=="object"):
            index = torch.where(mask > 0.5)
        else:
            print("""
            boostObject 的类型支持的增强类型为
            
                "object"
                "background"
            """)


        img[index]=boostTensor[index]

        return img






if __name__ == '__main__':
    train = FileDataSet("./data", True)
    for c in train:
        print(c[0].shape)
        print(c[1].shape)
        # chw->hwc
        pic = c[0].permute(1, 2, 0).contiguous().numpy()

        print(c[0][1])
        print(c[1])
        cv2.imshow("s", pic)
        cv2.waitKey(0)

    cv2.destroyAllWindows()