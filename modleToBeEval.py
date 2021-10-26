
"""
该部分为第三章测试的所有模型。
新增测试模型只需要在字典里添加即可
"""


from modle import myCNN         # 自定义的简单卷积神经网络
from torchvision.models import alexnet,\
                               resnet18,resnet34,resnet50,resnet101,\
                               mobilenet_v2,\
                               shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0,\
                               inception_v3,\
                               vgg11,vgg16,\
                               squeezenet1_0,squeezenet1_1,\
                               densenet121,densenet161,\
                               googlenet,\
                               mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3


"""
from .squeezenet import *
from .densenet import *
from .googlenet import *
from .mnasnet import *
"""

"""
暂时没有优化，随着模型的增加，内存会越来越大
toEval={
    'alexnet':alexnet(pretrained=False, progress=False,num_classes=10),
    'resnet18': resnet18(pretrained=False, progress=False,num_classes=10),
    'resnet34': resnet34(pretrained=False, progress=False,num_classes=10),
    'resnet50': resnet50(pretrained=False, progress=False,num_classes=10),
    'resnet101': resnet101(pretrained=False, progress=False,num_classes=10),
    'mobilenet_v2': mobilenet_v2(pretrained=False, progress=False,num_classes=10),
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5(pretrained=False, progress=False,num_classes=10),
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0(pretrained=False, progress=False,num_classes=10),
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5(pretrained=False, progress=False,num_classes=10),
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0(pretrained=False, progress=False,num_classes=10),
    'inception_v3': inception_v3(pretrained=False, progress=False,num_classes=10),
    'vgg11': vgg11(pretrained=False, progress=False,num_classes=10),
    'vgg16': vgg16(pretrained=False, progress=False,num_classes=10),
}
"""

modleFunBrk={


    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    #'resnet101': resnet101,
    'mobilenet_v2': mobilenet_v2,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,

    # 新加的
    #'squeezenet1_0':squeezenet1_0,  #这个莫名奇妙，原始的minst都不行？
    #'squeezenet1_1':squeezenet1_1,

    #'googlenet':googlenet,             ，报错，输出的结果没有size属性，导致pytorch内部报错，内部bug？
    'mnasnet0_5':mnasnet0_5,
    'mnasnet0_75':mnasnet0_75,
    'mnasnet1_0':mnasnet1_0,
    'mnasnet1_3':mnasnet1_3,
    #'densenet121':densenet121,
    #'inception_v3': inception_v3,              这三个模型必须要放大图像才行，这个单独测试
    #'vgg11': vgg11,
    #'vgg16': vgg16,
    #'alexnet': alexnet
}



modleFun={

    #'vgg11': vgg11,
    #'vgg16': vgg16,
    'alexnet': alexnet,
    #'densenet121':densenet121,
    #'densenet161':densenet161,

}



if __name__ == '__main__':
    import torch
    data=torch.randn(1,3,80,80)
    for k in modleFun:
        m = modleFun[k](pretrained=False, progress=False,num_classes=10)
        m = m.train()  # 必须是eval模式
        out = m(data)
        print(out)
        print(k)
        print(out.shape)
        #print(out[0])

    """


    f=[alexnet,\
                               resnet18,resnet34,resnet50,resnet101,\
                               mobilenet_v2,\
                               shufflenet_v2_x0_5,shufflenet_v2_x1_0,shufflenet_v2_x1_5,shufflenet_v2_x2_0,\
                               inception_v3,\
                               vgg11,vgg16,\
                               squeezenet1_0,squeezenet1_1,\
                               densenet121,\
                               googlenet,\
                               mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3]
    
    for m in f:
        print("'"+m.__name__+"'"+":"+m.__name__+",")
    """
