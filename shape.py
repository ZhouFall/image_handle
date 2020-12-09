# coding:utf-8
from PIL import Image, ImageDraw
import math

def isWhite(color):
    # 查看像素是否为白色
    if type(color) == type(1):
        light = color
    else:
        light = (color[0] + color[1] + color[2]) / 3
    if light < 150:
        return False
    return True

def getSharp(infile, outfile):
    # 获取图片轮廓
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

#厉害啊，老哥
# getSharp("photo.jfif", "2.png")
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

def merge_two_pic():
    img1 = Image.open('merge.png').convert("RGBA")#图片1
    img2 = Image.open('white.png').convert("RGBA")#图片2
    img2 = img2.resize((960, 960))
    # print(img1.size, img2.size)
    #取两张图片中最小的图片的像素
    new_x = min(img1.size, img2.size)[0]
    new_y = min(img1.size, img2.size)[1]
    new_img1 = cut_img(img1, new_x, new_y)
    new_img2 = cut_img(img2, new_x, new_y)
    # print(new_img1.size, new_img2.size)
    # #进行图片重叠  最后一个参数是图片的权值
    final_img2 = Image.blend(new_img1, new_img2, (math.sqrt(5)-1)/2)
    # final_img2 = Image.blend(new_img1, new_img2,  0.6)
    # #别问我为什么是  (math.sqrt(5)-1)/2   这个是黄金比例，哈哈！！
    final_img2.save('a5.png')
    # final_img2.show()

if __name__ == '__main__':
    # getSharp("photo.jfif", "2.png")
    # getSharp("alice.jpg", "alice_shape.png")
    merge_two_pic()