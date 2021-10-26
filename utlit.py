
import torch
import torch.nn as nn
import os

# 模型保存

class CNN():
    def __init__(self):
        print("开始加载训练好的模型")

myCNN=CNN()

'''传入的myCNN是一个实例化对象'''
def saveModel(Net=myCNN,path="./model",info=""):
    """
    myCNN不是模型定义的那个类
    而是，模型类的一个实例化对象
    """
    if(os.path.exists(path)):
        pass
    else:
        os.mkdir(path)
    path=path+'/'+info+'Model_params_only.pkl'
    torch.save(Net.state_dict(),path)

def restoreModel(Net=myCNN,path="./model",info=""):
    """
    这个myCNN是 模型定义的那个class 的一个实例化

    传进来的当指针来用


    将模型加载到 ret中
    其实，这个函数不用写也行。本身就很简单。 此时的ret就可以直接进行预测了。
    """
    ret=Net
    if (os.path.exists(path)):
        net_params=path+'/'+info+'Model_params_only.pkl'
        ret.load_state_dict(
            torch.load(net_params)
        )
        print("模型加载成功")
    else:
        print("加载模型参数失败，是不是保存出错了")

    return ret

