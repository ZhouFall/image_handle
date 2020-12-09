import PIL.Image as Image


# 以第一个像素为准，相同色改为透明
def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0))
    print(img.getpixel((0,0)))
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
    return img

if __name__ == '__main__':
    #此方法可以更改透明度
    img=Image.open('white.png')
    img=transparent_back(img)
    img.save('mask255.png')