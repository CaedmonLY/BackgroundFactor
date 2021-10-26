

import torch

class randBoost(torch.nn.Module):
    """
    数据增强，继承nn.Module，以便可以在gpu上运行                                         cpu上也能执行

    目前不打算将它纳入到DataSet类里面处理。     直接对一个gpu中的一个batch处理
    这样的话，和以往框架有点不同的时，数据加载完后，还需进行    “再处理”                       直接在cpu上怕有点慢


    输入一个“预处理后的图像”和一个对象mask。
    对象以外的背景用0均值随机数替代。

    """

    def __call__(self,batchimg_tensor,batchmask_tensor):
        """
        batchimg :   [batchsize,3,h,w]         float
        btachmask:   [batch,h,w]               float

        return  :    [batch,3,h,w]
        """
        batchimg=batchimg_tensor.float()
        batchmask=batchmask_tensor.float()
        batchmask = torch.stack((batchmask, batchmask, batchmask), 1)      # 也构造成三通道，索引的时候直接和img匹配

        shape=batchimg.shape
        if(torch.cuda.is_available()):
            boostTensor=torch.randn(shape).cuda()               # 标准正态分布中，大于3.9的部分，概率基本为0了。可以近似认为抽出来的数大致在[-3.9,3.9]
        else:
            boostTensor = torch.randn(shape)
        #boostTensor=boostTensor/3.9
        boostTensor = boostTensor / 3.9



        """        
        print("debug boost")
        print(batchimg.shape)
        print(batchmask.shape)
        print(boostTensor.shape)
        """

        #为了本地测试，传进来的数是   [batch,h,w,c],[batch,h,w]      *255是为了cv2演示效果
        #boostTensor=boostTensor*255
        #boostTensor=torch.abs(boostTensor)

        background=torch.where(batchmask<0.5)         #  float数，还是确定一个范围比较保险，选出 ==0 的index


        batchimg[background]=boostTensor[background]

        return batchimg
