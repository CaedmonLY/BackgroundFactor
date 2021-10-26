"""
独立完成一个最简单的模型
用于分类mnist


模型计算只需要数据就行，传进来的只是图像数据，标签的话，在计算损失的时候采用到

另外，pytorch的卷积核是可以直接支持 多批次处理的，即[batchsize,28,28,3]，经过卷积核计算出来的也是batch数据




毋文靖
2019/12/10

"""

import torch
import torch.nn as nn
import torch.nn.functional as f



class myCNN(nn.Module):

    def __init__(self):

        super(myCNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=12,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            #            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(12, 12, 5),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation              # output shape (32, 7, 7)
        )
        self.conv3 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(12, 12, 3),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation              # output shape (32, 7, 7)
        )

        self.fc1=nn.Linear(16*16*12,512)
        self.fc2=nn.Linear(512,10)

        self.softmax=nn.Softmax(dim=1)



    def forward(self,data):
        """
        pytorch 默认的是 CHW

        data: [batchsize,3,28,28]
        """
        batchsize,c,h,w=data.size()

        x=self.conv1(data)

        x=self.conv2(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.conv3(x)

        x=x.reshape(batchsize,-1)

        x=self.fc1(x)
        x=self.fc2(x)

        x=self.softmax(x).float()

        return x

        #index0 = torch.arange(batchsize)
        #index1=torch.argmax(x,dim=1)

        # 这样训练貌似很差吧，还是输出成  10位的编码。 当然，图片的lable也要编码为10位。   但是，用一个二分类就分类为  [-1、1]就行了。两位编码
        #return x[index0,index1]



if __name__ == '__main__':
    conv1 = nn.Conv2d(3, 12, 3, 1, 1)
    conv2 = nn.Conv2d(12, 12, 5 )
    conv3 = nn.Conv2d(12, 12, 3 )
    x=torch.ones((5,3,28,28))
    y=conv1(x)    #28*28
    y=conv2(y)    #24*24
    y=conv3(y)    #22*22

    #print(y.size())

    mod=myCNN()
    y=mod.forward(x)
    print(y)

    print("data")
    data=torch.arange(4*2).reshape(4,2)
    print(data)
    a=[3]
    b=[0]
    print(data[a,b])





















