# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-04

"""
使用SURF特征提取提取图片特征

改进代码：
1.featureExtract当写入文件失败时return false,抛出写入文件错误

修改问题：
1. 读取文件中图片失败：
    图片路径包含中文
2. 写入SURF描述矩阵失败：
    写入的SURF特征描述矩阵的desNdarray,KeyNumList名字（str格式）与变量名重合，将变量desNdarray,KeyNumList修改为desNdarray_name,KeyNumList_name
3. 使用csv格式的文件写出
4. 一些图片不能检测出surf局部特征（des is None)
5. csv.writerow()需要的参数是可迭代对象，也就是说输入的列表必须是二维的。————>将特征数量和标签值重合，写入一个可迭代对象
6. ../表示上一级路径

优化思考：
1. 进行轮廓边缘检测再进行轮廓的特征提取？（因为物品上的动画图案可能会使训练过拟合，需要更加注重于物品的轮廓特征！）——先进行Canny边缘检测？ 有什么轮廓特征提取的方法？
2. 剔除掉没有特征提取到的图片？？？
3. 在这里就划分好训练集和测试集（不能使用sklearn的划分函数，因为其X特征矩阵维度与label维度不一致）
"""

import cv2 as cv
import numpy as np
import os
import csv
import time


# 对一个文件夹内的图片都进行SURF特征提取,同时将对图片的SURF特征提取值直接以数组形式保存在文件中，可直接利用np.loadtxt()读取数据，返回的为ndarray(float组成的数据！）
# 输入：图片文件夹路径：images_folder
# 输出：存储图片的特征描述矩阵文件：desNdarray, 每一副图片的特征点个数列表：KeyNumList,同时将信息存储下来（详情见readme.txt)
def featureExtract(images_folder):
    """
    对一个文件夹内的图片都进行SURF特征提取,同时将对图片的SURF特征提取值直接以数组形式保存在csv文件中
    :param images_folder: 图片文件夹路径
    :return: （训练集&测试集)存储图片的特征描述矩阵文件：desNdarray, 每一副图片的特征点个数列表：KeyNumList;同时将信息存储下来（详情见readme.txt)
    """
    start_time = time.time()
    file_label_list = getFileLabelList(images_folder)
    # 训练集数据
    KeyNumList_train = []
    desNdarray_train = []
    # 测试集数据
    KeyNumList_test = []
    desNdarray_test = []
    ratio = 0   #比例选择为0.2，每5张选一张
    SURF_threshhold = 1000
    None_num = 0
    Nlist = []
    for folder_name in file_label_list:  # 遍历每一个文件夹
        folder_path = images_folder + folder_name + '/'
        for filename in os.listdir(folder_path):  # 遍历图片目录的每一幅图片
            if '.jpg' in filename:
                ratio+=1
                filePath = folder_path + filename
                img = cv.imread(filename=filePath)
                blurred = cv.GaussianBlur(img, (3, 3), 0)  # 高斯模糊降噪
                gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)  # 转换成灰度图像

                detector = cv.xfeatures2d.SURF_create(SURF_threshhold)  # 创建SURF
                kps, des = detector.detectAndCompute(gray, None)  # 进行SURF特征提取，返回图片特征的关键点和描述符

                # 展示SURF在图像上的特征点
                # img2 = img.copy()
                # img2 = cv.drawKeypoints(image=img, outImage=img2, keypoints=kps, color=(255, 0, 0))  # 绘制关键点
                # print(type(img2))
                # cv.imshow('kps',img2)
                # cv.waitKey(0)
                # print(np.shape(des))
                if ratio%5 == 0:
                    if des is None:
                        None_num+=1
                        Nlist.append(filename)  # 存下没有SURF特征的图片名字
                        continue
                    else:
                        desNdarray_test.extend(des)
                        KeyNumList_test.append([len(kps), folder_name])  # 整合每幅图片包含的关键点个数
                else:
                    if des is None:  # 存在有些图片没有SURF特征的情况
                        # KeyNumList_train.append([0, folder_name])
                        None_num += 1  # 记录没有SURF特征的图片的数量
                        Nlist.append(filename)  # 存下没有SURF特征的图片名字
                        # 剔除掉没有特征的文件
                        continue
                    else:
                        desNdarray_train.extend(des)  # 整合描述符(存储为2维list）
                        KeyNumList_train.append([len(kps), folder_name])  # 整合每幅图片包含的关键点个数
            else:
                continue
        print('文件夹' + folder_name + '提取完毕')
    end_time = time.time()
    csvWrite('../result/SURF_threshhold' + str(SURF_threshhold) + 'des_train' + '.csv', desNdarray_train)
    csvWrite('../result/SURF_threshhold' + str(SURF_threshhold) + 'label_train' + '.csv', KeyNumList_train)
    csvWrite('../result/SURF_threshhold' + str(SURF_threshhold) + 'des_test' + '.csv', desNdarray_test)
    csvWrite('../result/SURF_threshhold' + str(SURF_threshhold) + 'label_test' + '.csv', KeyNumList_test)
    f = open('../result/SURF_record.txt', 'a+')
    f.write('SURF的阈值为:%d'%SURF_threshhold+'\n')
    f.write('用时%.8ss' % (end_time - start_time) + '\n')
    f.write('图片样本总数量为%d' % ratio + '\n')
    f.write('训练集特征大小:%d'%np.shape(desNdarray_train)[0]+' '+'训练集大小:%d'%np.shape(KeyNumList_train)[0]+'\n')
    f.write('测试集特征大小:%d' % np.shape(desNdarray_test)[0] + ' ' + '测试集大小:%d' % np.shape(KeyNumList_test)[0] + '\n')
    f.write('没有SURF特征的图片数量为：%d,文件名如下：' % None_num + '\n')
    for name in Nlist:
        f.write(name + ';')
    f.write('\r\n')
    f.close()
    print('没有SURF特征的图片数量为：%d' % None_num)
    print('特征提取完毕')
    print('用时%.8s' % (end_time - start_time) + '\r\n')

    return desNdarray_train, KeyNumList_train,desNdarray_test,KeyNumList_test


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


# 获得文件夹标签索引
# 输入：图片文件夹路径:path
# 输出：文件夹标签索引:file_label_list
def getFileLabelList(path):
    file_label_list = []
    for filename in os.listdir(path):
        if filename not in file_label_list:
            file_label_list.append(filename)
        else:
            raise NameError('文件夹命名错误')
    return file_label_list


if __name__ == '__main__':
    images_folder = 'D:/MachineLearning_DataSet/DTD_DescrubableTextures/dtd/images/'  # 尝试给出图片文件夹
    # images_folder = '../temp/'  # 该文件位于文件夹中，故需要使用..表示上一级文件夹
    # OutputPath = './imageInfo/'  # 给出保存文件路径
    featureExtract(images_folder=images_folder)
    # 尝试读出数据
    # desNdarray = np.loadtxt(images_folder + 'desNdarray')
    # print(np.shape(desNdarray))
    # KeyNumList = np.loadtxt(images_folder + 'KeyNumList')
    # print(np.shape(KeyNumList))
