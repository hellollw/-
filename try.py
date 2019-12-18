import cv2 as cv
import numpy as np
import os
import csv

# 使用csv写入文件(一维列表对应一行）
# 输入：文件名:dataname, 列表数据:datalist
# 输出：在指定位置处写入指定姓名的文件
def csvWrite(dataname, datalist):
    f = open(dataname, 'w', encoding='utf-8', newline='')  # 设置newline=''，不会产生空行
    csv_writer = csv.writer(f)
    for cur_data in datalist:  # datalist应为二维数组
        csv_writer.writerow(cur_data)
    f.close()
    print('写出' + dataname + '成功')


# 定义生成器，实现增量式学习，防止内存溢出的错误
# 输入:数据集位置:data_path,循环读取的次数:echos,最小训练集数量:minibatch_size
# 返回：一个生成器
def dataGenerator(data_path,echos, minibatch_size=1000):
    """
    定义生成器，实现增量式学习，防止内存溢出的错误
    :param data_path: 数据集位置
    :param minibatch_size: 最小训练集数量
    :return:返回一个生成器，一次初始化后相当于一个可迭代变量
    """
    X = []
    cur_line_num = 0


    for j in range(echos):
        csvfile = open(data_path, 'r')
        reader = csv.reader(csvfile)
        for line in reader:
            X.append(line)  # 这里要将数据转化成float类型
            cur_line_num += 1
            if cur_line_num >= minibatch_size:  # 当生成器当前生成的数据数量大于设定的最小数量大小时
                # X, y = np.array(X), np.array(y)  # 将数据转成numpy的array类型并返回
                yield X
                X = []
                cur_line_num = 0
        csvfile.close()


if __name__ =='__main__':
    SURF_threshold = 2000
    filePath1 = 'D:/MachineLearning_DataSet/DTD_DescrubableTextures/dtd/images/blotchy/blotchy_0029.jpg'
    filePath2 = 'D:/MachineLearning_DataSet/DTD_DescrubableTextures/dtd/images/crystalline/crystalline_0124.jpg'
    img = cv.imread(filename=filePath1)
    blurred = cv.GaussianBlur(img, (3, 3), 0)  # 高斯模糊降噪
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)  # 转换成灰度图像

    detector = cv.xfeatures2d.SURF_create(SURF_threshold)  # 创建SURF
    kps, des = detector.detectAndCompute(gray, None)  # 进行SURF特征提取，返回图片特征的关键点和描述符

    # 展示SURF在图像上的特征点
    img2 = img.copy()
    img2 = cv.drawKeypoints(image=img, outImage=img2, keypoints=kps, color=(255, 0, 0))  # 绘制关键点
    print(type(img2))
    cv.imshow('kps',img2)
    cv.waitKey(0)



