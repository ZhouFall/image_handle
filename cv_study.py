# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image

def firt_handle():
   # IMAGE_NAME = "mask.png"
   IMAGE_NAME = "alice.jpg"
   SAVE_IMAGE_NAME = "canny_"+IMAGE_NAME
   img = cv2.imread(IMAGE_NAME)
   img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   c_canny_img = cv2.Canny(img2gray,50,150)
   cv2.imwrite('canny_alice.png', c_canny_img)
   # cv2.imshow('mask',c_canny_img)
   k = cv2.waitKey(500) & 0xFF
   if k == 27:
      cv2.destroyAllWindows()
   #canny算法，强啊，牛逼

def get_bordor():
   # img = cv2.imread('canny_mask.png')
   img = cv2.imread('canny_alice.png')
   rows, cols, ch = img.shape
   SIZE = 3  # 卷积核大小
   P = int(SIZE / 2)
   BLACK = [0, 0, 0]
   WHITE = [255, 255, 255]
   BEGIN = False
   BP = []

   for row in range(P, rows - P, 1):
      for col in range(P, cols - P, 1):
         # print(img[row,col])
         if (img[row, col] == WHITE).all():
            kernal = []
            for i in range(row - P, row + P + 1, 1):
               for j in range(col - P, col + P + 1, 1):
                  kernal.append(img[i, j])
                  if (img[i, j] == BLACK).all():
                     # print(i,j)
                     BP.append([i, j])

   print(len(BP))
   uniqueBP = np.array(list(set([tuple(c) for c in BP])))
   print(len(uniqueBP))

   for x, y in uniqueBP:
      img[x, y] = WHITE
   # cv2.imshow('img', img)
   cv2.imwrite('canny_alice2.png', img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()


def cv_merge():
   img = cv2.imread('merge.png')
   # img2 = Image.open('mask2.png').convert("RGB")
   img2 = Image.open('white.png')
   # img2 = Image.open('mask.png')
   out = img2.resize((960,960))
   out.save('resize.png')
   img2 = cv2.imread('resize.png')
   print(img.shape)
   print(img2.shape)

   img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
   print(img2gray.shape)
   mask = cv2.bitwise_and(img, img2, mask=img2gray)
   # mask = cv2.bitwise_not(img, img2, mask=img2gray)
   cv2.imwrite('canny_merge.png', mask)
   # cv2.imshow('mask', mask)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

if __name__ == "__main__":
   # firt_handle()
   # get_bordor()
   # get_bordor()
   cv_merge()