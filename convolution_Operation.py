#https://blog.csdn.net/CSDN_CQD/article/details/105224388

import numpy as np
def zero_pad(x,pad_height,pad_width):#先在待处理矩阵周围填充0
   H ,W = x.shape #待处理矩阵的尺寸
   out = np.zeros((H+2*pad_height,W+2*pad_width)) #知道尺寸后先全填入0
   out[pad_height:H+pad_height,pad_width:W+pad_width] = x#后在中间填入x，这样边缘填充0就完成了
   return out  

def conv_fast(x,h):
  Hi, Wi = x.shape#待处理矩阵的尺寸

  Hh, Wh = h.shape#卷积核的尺寸
  out = np.zeros((Hi,Wi))#提前经过计算得到输入矩阵的大小，先用0填充
  pad_height = Hh // 2#mode为same情况下，填充0的数量取决于卷积核h的尺寸   https://www.cnblogs.com/sddai/p/10512784.html
  pad_width = Wh // 2
  image_padding = zero_pad(x,pad_height,pad_width)
  h_flip = np.flip(np.flip(h,0),1)    #np.flip 是翻转函数,参数0为上下翻转也就是行翻转,而参数1为左右翻转也就是列翻转
  for i in range(Hi):
    for j in range(Wi):
      out[i][j] = np.sum(np.multiply(h_flip,image_padding[i:i+Hh,j:j+Wh]))   #加权求和后写入结果到out对应位置     
  return out
  
x = np.array([[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]])
h = np.array([[-1,-2,-1],
     [0,0,0],
     [1,2,1]])

print(conv_fast(x,h))


