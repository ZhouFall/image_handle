#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 技术支持：https://www.jianshu.com/u/69f40328d4f0
# 技术支持 https://china-testing.github.io/
# https://github.com/china-testing/python-api-tesing/blob/master/practices/pil_merge.py
# 项目实战讨论QQ群630011153 144081101
# CreateDate: 2018-11-22

import math
from PIL import Image
import os
import cv2



#一共900个照片，拼成30*30

#照片原尺寸是320*320，缩放一下，变成32*32
width = 32
height = 32


# 查找当前路径下所有的html文件
def walkFile():
    for root, dirs, files in os.walk(r'pic'):
        for f in files:
            if f.endswith('.png') or f.endswith('.jpg'):
                fullname = os.path.join(root, f)
                yield fullname
                # yield f

list_im = []
size = (32, 32)
filename = walkFile()
for i in filename:
    # image_path = 'album\\' + i
    # img = Image.open(image_path)
    # out = img.resize(size)
    # out.save('pic\\'+i)
    # img = cv2.imread(image_path)
    # new_path = 'pic\\' + i
    # try:
    #     out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    #     cv2.imwrite(new_path, out)
    # except:
    #     print('{}这张图片不行'.format(image_path))
    list_im.append(i)
    # print(new_path)
    # break

print(list_im)


imgs = [Image.open(i) for i in list_im]
#math.cell用于向上取整
column = 30
row_num = math.ceil(len(imgs)/column)
# row_num = 30
print(height*row_num)
target = Image.new('RGB', (width*column, height*row_num))
for i in range(len(list_im)):
    if i % column == 0:
        end = len(list_im) if i + column > len(list_im) else i + column
        for col, image in enumerate(imgs[i:i+column]):
            target.paste(image, (width*col, height*(i//column),width*(col + 1), height*(i//column + 1)))
target.show()
target.save('merge.png')