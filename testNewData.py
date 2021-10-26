import cv2
import datetime

epoch=1.324234324234

f=open("test.txt",'w')
f.write("{:.1f},{:.2f}、{:.3f} {:.4f}\n".format(
epoch,epoch,epoch,epoch
))

import time
# 格式化成2020-12-01@21:44:07形式
now=time.strftime("%Y-%m-%d@%H:%M:%S", time.localtime())
print(now)

a=1
log = "epoch {:d} loss {:.6f} test_accuracy {:.5f}\n".format(
    a, epoch, epoch
)

print(log)

