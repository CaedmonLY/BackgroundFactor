import torch
import numpy as np



# 计算等位置，两个tensor相等的个数
a=torch.arange(10).reshape(2,5)
b=torch.arange(10).reshape(2,5)
ans=(a==b).float()
ans=torch.sum(ans)



# 列表索引
a=torch.zeros(25).reshape(5,5)

i=[1,2,4]
j=[1,2,4]

#print(a)
a[i,j]=1
#print(a)

#  argmax?
a=torch.arange(16*4).reshape(16,4)
print(a)
i=torch.arange(16)
index=torch.argmax(a,dim=1)
a[i,index]=0
print(a)


# cv 添加文字

import cv2
img = cv2.imread('test.jpg')
font = cv2.FONT_HERSHEY_SIMPLEX

imgzi = cv2.putText(img, '000', (50, 300), font, 1.2, (255, 255, 255), 2)
                   # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度



cv2.imshow('backup',imgzi)  # 显示原图像的备份
cv2.waitKey()
cv2.destroyAllWindows()


