#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 参考网站：https://www.jianshu.com/u/69f40328d4f0


import math
from PIL import Image,ImageDraw
import os
import cv2
import numpy as np

# 查找当前路径下所有的png文件
def walkFile(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.png') or f.endswith('.jpg'):
                fullname = os.path.join(root, f)
                yield fullname
                # yield f

#该函数的作用是由于 Image.blend()函数只能对像素大小一样的图片进行重叠，故需要对图片进行剪切。
def cut_img(img, x, y):
    """
    函数功能：进行图片裁剪（从中心点出发）
    :param img: 要裁剪的图片
    :param x: 需要裁剪的宽度
    :param y: 需要裁剪的高
    :return: 返回裁剪后的图片
    """
    x_center = img.size[0] / 2
    y_center = img.size[1] / 2
    new_x1 = x_center - x//2
    new_y1 = y_center - y//2
    new_x2 = x_center + x//2
    new_y2 = y_center + y//2
    new_img = img.crop((new_x1, new_y1, new_x2, new_y2))
    return new_img

def isWhite(color):
    # 查看像素是否为白色
    if type(color) == type(1):
        light = color
    else:
        light = (color[0] + color[1] + color[2]) / 3
    if light < 150:
        return False
    return True

class image_handle():
    def __init__(self):
        self.size = (32, 32)

    def image_resize(self,path):
        list_im = []
        #删掉多余的图片，只保留900张，拼接成30*30的大图
        filename = walkFile(path)
        for i in filename:
            ######方法一，采用image.resize这张方法，压缩之后像素会差一点
            # img = Image.open(i)
            # out = img.resize(self.size)
            # new_path = 'pic1\\'+str(i.split('\\')[1])
            # out.save(new_path)
            #####方法二，采用cv2.resize
            img = cv2.imread(i)
            new_path = 'pic\\'+str(i.split('\\')[1])
            try:
                #有一张图片比较特殊，resize会报错，加个try处理
                out = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_path, out)
            except:
                print('{}这张图片不行'.format(i))
            #######################################
            list_im.append(new_path)
        print(len(list_im))
        return list_im

    #  #将900张图片重新设置大小，再合并成一张30*30
    def image_merge(self,path):
        list_im = self.image_resize(path)
        imgs = [Image.open(i) for i in list_im]
        #math.cell用于向上取整
        column = 30
        row_num = math.ceil(len(imgs)/column)
        # row_num = 30
        # 照片原尺寸是320*320，缩放一下，变成32*32
        width = 32
        height = 32
        # print(height*row_num)    #32*30
        target = Image.new('RGB', (width*column, height*row_num))
        for i in range(len(list_im)):
            if i % column == 0:
                end = len(list_im) if i + column > len(list_im) else i + column
                for col, image in enumerate(imgs[i:i+column]):
                    target.paste(image, (width*col, height*(i//column),width*(col + 1), height*(i//column + 1)))
        # target.show()
        target.save(r'output/merge.png')

    #如果有需要，可以使用下面的方法，更改图像透明度，实现方法是更改图像的第四通道
    def change_channel4(self,image):
        # 此方法可以更改透明度
        img = Image.open(image)
        img = img.convert('RGBA')
        L, H = img.size
        #使用0，0点的像素作为标准，所有像素和0，0一样的都设置成透明,最好是处理过的01图像，不然需要加一个容错，和这个像素相差30以内的都设置透明
        color_0 = img.getpixel((0, 0))
        # print(img.getpixel((0, 0)))
        for h in range(H):
            for l in range(L):
                dot = (l, h)
                color_1 = img.getpixel(dot)
                if color_1 == color_0:
                    #(0,)，0是完全透明，255是完全不透明
                    color_1 = color_1[:-1] + (0,)
                    img.putpixel(dot, color_1)
        img.save('mask255.png')
        # return img

    # 获取图片轮廓方法一
    def get_shape_method1(self,infile,outfile):
        img = Image.open(infile)
        data = img.getdata()
        outimg = Image.new("1", img.size)
        drawimg = ImageDraw.Draw(outimg)
        width, height = img.size
        cnt = 0
        for y in range(0, height):
            for x in range(0, width):
                drawimg.point((x, y), 255)
                if isWhite(data[y * width + x]):
                    continue
                if y == 0 or x == 0 or y == height - 1 or x == width - 1:
                    continue
                if not isWhite(data[(y - 1) * width + x]) and not isWhite(data[(y + 1) * width + x]):
                    if not isWhite(data[y * width + (x - 1)]) and not isWhite(data[y * width + x + 1]):
                        continue
                cnt += 1
                drawimg.point((x, y), 0)
        outimg.save(outfile)
        print(cnt)

    def get_shape_canny1(self,infile,outfile):
        # 读取原灰度图片
        image = cv2.imread(infile)
        # cv2.imshow("image", image)  # 将原图片命名为“image”显示出来
        # 图像的阈值分割处理，即将图像处理成非黑即白的二值图像
        ret, image1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)  # binary（黑白二值），ret代表阈值，80是低阈值，255是高阈值
        # cv2.imshow('image1', image1)  # 将阈值分割处理后的图片命名为“image1”显示出来
        cv2.imwrite('image1.png', image1)
        # 二值图像的反色处理，将图片像素取反
        height, width, channel = image1.shape  # 返回图片大小
        image2 = image1.copy()
        for i in range(height):
            for j in range(width):
                image2[i, j] = (255 - image1[i, j])
        # cv2.imshow('image2', image2)  # 将反色处理后的图片命名为“image2”显示出来
        cv2.imwrite('image2.png', image2)
        # 边缘提取，使用Canny函数
        image3 = cv2.Canny(image2, 80, 255)  # 设置80为低阈值，255为高阈值
        # cv2.imshow('image3', image2_3)  # 将边缘提取后的图片命名为“image2_3”显示出来
        cv2.imwrite('image3.png', image3)

        # 再次对图像进行反色处理使提取的边缘为黑色，其余部分为白色，方法同image2
        # height1, width1, channel = image3.shape
        height1, width1 = image3.shape
        image4 = image3.copy()
        for i in range(height1):
            for j in range(width1):
                image4[i, j] = (255 - image3[i, j])
                # 加两print，成功实现
                # print(image3[i,j])
                # 以下为尝试，不要那么白，这样合成的时候不会太难看；还是要白点好，太黑了看不清
                # if image3[i,j][0] == 255:
                #     image3[i, j] = [180,180,180]
        # cv2.imshow('image4', image4)#将边缘提取后反色处理的图片命名为“image4”显示出来
        cv2.imwrite(outfile, image4)
        # cv2.waitKey(0)  # 等待键盘输入，不输入则无限等待
        # cv2.destroyAllWindows()  # 销毁所有窗口

    # canny算法，强啊，牛逼
    def get_shape_canny2(self, infile, outfile):
        img = cv2.imread(infile)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(img2gray, 50, 150)
        cv2.imwrite('canny_interal.png', canny_img)
        canny_img = cv2.imread('canny_interal.png')
        rows, cols, ch = canny_img.shape
        SIZE = 3  # 卷积核大小
        P = int(SIZE / 2)
        BLACK = [0, 0, 0]
        WHITE = [255, 255, 255]
        # BEGIN = False
        BP = []
        for row in range(P, rows - P, 1):
            for col in range(P, cols - P, 1):
                # print(img[row,col])
                if (canny_img[row, col] == WHITE).all():
                    kernal = []
                    for i in range(row - P, row + P + 1, 1):
                        for j in range(col - P, col + P + 1, 1):
                            kernal.append(canny_img[i, j])
                            if (img[i, j] == BLACK).all():
                                # print(i,j)
                                BP.append([i, j])
        #print(len(BP))   print(len(uniqueBP))
        uniqueBP = np.array(list(set([tuple(c) for c in BP])))
        for x, y in uniqueBP:
            canny_img[x, y] = WHITE
        # cv2.imwrite(outfile, canny_img)
        #图片取反，变成白底，黑线条
        height1, width1,ch = canny_img.shape
        image4 = canny_img.copy()
        for i in range(height1):
            for j in range(width1):
                image4[i, j] = (255 - canny_img[i, j])
        cv2.imwrite(outfile, image4)

    def combine_two_image_cv2(self,image1,image2,image3):
        #视图片情况考虑需不需要添加.covert("RGB")
        # img2 = Image.open('mask2.png').convert("RGB")
        # img2 = Image.open(image2)
        # out = img2.resize((960, 960))
        # out.save('resize.png')
        # img2 = cv2.imread('resize.png')
        img1 = cv2.imread(image1)
        img2 = cv2.imread(image2)
        self.size = (960, 960)
        img1 = cv2.resize(img1, self.size, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, self.size, interpolation=cv2.INTER_AREA)
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        print(img2gray.shape)
        mask = cv2.bitwise_and(img1, img2, mask=img2gray)
        cv2.imwrite(image3, mask)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def combine_two_image_pil(self,image1,image2,image3):
        img1 = Image.open(image1).convert("RGBA")  # 图片1
        img2 = Image.open(image2).convert("RGBA")  # 图片2
        img2 = img2.resize((960, 960))
        # 取两张图片中最小的图片的像素
        new_x = min(img1.size, img2.size)[0]
        new_y = min(img1.size, img2.size)[1]
        print(new_x)
        #cut_img的作用是将两张图片大小设置成一致，可以采用resize的方法。
        new_img1 = cut_img(img1, new_x, new_y)
        new_img2 = cut_img(img2, new_x, new_y)
        # #进行图片重叠  最后一个参数是图片的权值
        final_img2 = Image.blend(new_img1, new_img2, (math.sqrt(5) - 1) / 2)
        final_img2.save(image3)
        # final_img2.show()


if __name__ == '__main__' :
    img = image_handle()
    # img.image_merge(r'album')
    # img.get_shape_method1("photo.jfif", "method1.png")
    # img.get_shape_canny1("photo.jfif", "canny1.png")    #canny1获取轮廓的效果比method1要好
    # img.get_shape_canny2("photo.jfif", "canny2.png")    #canny2获取轮廓效果最好
    # img.change_channel4(r'canny2.png')
    img.combine_two_image_pil('merge.png','white.png','pil_merge1.png')
    img.combine_two_image_cv2('merge.png','white.png','canny_merge1.png')