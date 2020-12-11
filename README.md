# image_handle

##### 工程说明
1. 将从网易云下载下来的1000张图片进行拼接
2. 排版设想是30*30，总共使用900张图片，去掉多余的图片
3. 一开始计划直接排成指定形状，不过没找到现成的方法，转变思路，改为先拼接成一个正方体，再加一个mask
4. 拼接的时候需要把大小改成一致，先将图片resize一下，然后拼接，结果参考data/merge.png

5. 接下来是mask的处理，选择了阿七这种图片，轮廓清晰，但是线条不是很好，抠出来的图片不够饱满。可以考虑选择爱心之类的形状
6. pil和cv2都有处理线条轮廓的方法，cv2主要使用canny算法，参考了两个博主的写法，效果不同，这部分需要理解一下算法的原理
7. 获得轮廓之后，将图像取反一下，线条是黑色的，其他部分是白色的，参考output/image3
8. image3和merge.png合并效果不太好，于是释放大招，使用图像编辑器把中间给扣空了，参照output/white图片
9. 合并采用了两种方式，一种是跟mask相与，另一种是加权叠加，结果分别是cv2_merge 和pil_merge
***
##### 参考链接
1. https://blog.csdn.net/ngsb153/article/details/105707492 opencv-python 将图片批量压缩为指定的尺寸
2. https://blog.csdn.net/dream_people/article/details/83372354 如何用python取图片轮廓
3. https://blog.csdn.net/septwolves2015/article/details/97896681?utm_medium=distribute.pc_relevant.none-task-blog-baidulandingword-3&spm=1001.2101.3001.4242 \
python-opencv边缘清洗法提取图片轮廓和前景内容
4. https://www.jianshu.com/p/c04e34883c18 Python PIL实现图片重叠
5. https://blog.csdn.net/qq_40878431/article/details/82941982 Python PIL.Image之修改图片背景为透明
6. https://blog.csdn.net/guduruyu/article/details/71439733?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control \
【python图像处理】两幅图像的合成一幅图像（blending two images）
