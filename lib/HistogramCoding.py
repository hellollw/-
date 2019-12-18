# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-05

"""
采用histogram（频率统计直方图编码）对每一个图片样本进行编码
采用SVM实现多分类

修改：
    1. 如何让生成器读完文件（最后一步时不产生遗漏）——使用死方法，直接定义好样本总数
    2. 是否需要归一化？？ 尝试将特征向量归一化(归一化还是牛逼的） 训练准确率，测试准确率提升了5%

histogram编码流程：
    初始化总样本编码特征空间
    对每一幅样本图：
        新建该幅图的n_cluster维度向量
        读取该幅图对应的SURF特征数量
        从上一幅图的特征数量结尾数值到该幅图的特征数量结尾数值:
            读取该特征对应的分类簇中心
            该幅图的n_cluster维度向量对应位置增1
        将该幅图的n_cluster维度向量放入总样本特征空间中
"""

import numpy as np
import csv
import time
import joblib
from sklearn.cluster import MiniBatchKMeans


# 定义生成器，实现增量式学习，防止内存溢出的错误
# 输入:数据集位置:data_path, 需要读取的文件信息:SURF_threshold, 循环读取次数:echos,最小训练集数量:minibatch_size
# 返回：一个生成器
def dataGenerator(data_path, SURF_threshold, echos, minibatch_size=5000):
    """
    定义生成器，实现增量式学习，防止内存溢出的错误
    :param data_path: 数据集位置
    :param minibatch_size: 最小训练集数量
    :return:返回一个生成器，一次初始化后相当于一个可迭代变量
    """
    X = []
    cur_line_num = 0

    # 定义训练集和测试集特征总数量
    training_num = 2945430
    test_num = 716910
    try_num = 53888

    for j in range(echos):
        f1 = open(data_path + 'SURF_threshhold' + str(SURF_threshold) + 'des_train' + '.csv', 'r')
        reader = csv.reader(f1)
        cur_index = 1
        for line in reader:
            X.append(line)  # 这里要将数据转化成float类型
            cur_line_num += 1
            if cur_line_num >= minibatch_size:  # 当生成器当前生成的数据数量大于设定的最小数量大小时
                X_float = str2float(X)
                yield X_float
                X = []
                cur_line_num = 0
            if cur_index == training_num:  # 到最低部
                X_float = str2float(X)
                yield X_float
            cur_index = cur_index + 1
        f1.close()


def getWholelabel(labelstring):
    wholelabel = []
    for label in labelstring:
        if label not in wholelabel:
            wholelabel.append(label)
        else:
            continue
    csvWrite('../result/coding/wholelabel.csv', enumerate(wholelabel))
    print('wholelabel写出成功%d '%len(wholelabel))
    return wholelabel


# 将二维字符串型数组转换为浮点型
# 输入：二维字符串数组:datastr
# 输出：浮点数组:datafloat
def str2float(datastr):
    m, n = np.shape(datastr)
    datafloat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            try:
                datafloat[i][j] = float(datastr[i][j].rstrip())  # 数组遍历实现转换
            except:
                print('为0位置在于%d行%d列' % (i + 1, j + 1))
                datafloat[i][j] = 0.0
    return datafloat


# 使用csv写入文件
# 输入：文件名:dataname, 列表数据:datalist
# 输出：在指定位置处写入指定姓名的文件
def csvWrite(dataname, datalist):
    f = open(dataname, 'w', encoding='utf-8', newline='')  # 设置newline=''，不会产生空行
    csv_writer = csv.writer(f)
    for cur_data in datalist:  # datalist应为二维数组
        csv_writer.writerow(cur_data)
    f.close()
    print('写出' + dataname + '成功')


# 将标签字符串文件转换为数字文件(符合sklearn的要求）
# 输入：字符串样本集:labelstring, 整体标签名称：Wholelabel
# 输出：转换为对应的数字labelint, 集和种类：labelnum
def string2int(labelstring, wholelabel):
    labelint = []
    for cur_label in labelstring:
        label_index = wholelabel.index(cur_label)
        labelint.append(label_index)  # 转换为对应的种类数字
    return labelint

# 获得对应特征的标签值
def getHistogramlabel(image_feature,wholelabel):
    """

    :param image_feature:每一幅图片对应的字符串标签
    :param wholelabel: 编码的总标签
    :return: 编码空间的标签值
    """
    histogram_label = string2int(image_feature, wholelabel)
    print('获得样本标签完成')
    return histogram_label

# histogram编码
# 输入：SURF特征的簇分类标签：SURF_cluster,每一副样本图片所包含的特征数量:image_feature, 簇中心数量
# 输出：样本编码特征空间:histogram_code_space
def getHistogramCode(SURF_cluster, image_feature, n_cluster):
    histogram_code_space = []
    last_index = 0  # 上一次读取到的特征位置
    for cur_image_str in image_feature:  # 第一列存放每幅图片所包含的特征数量
        cur_image_num = int(cur_image_str)
        cur_image_vector = np.zeros((1, n_cluster))
        for cur_SURF_index in range(last_index, last_index + cur_image_num):
            cur_label = SURF_cluster[cur_SURF_index]
            cur_image_vector[0,cur_label] += 1
        cur_image_vector = cur_image_vector/np.sum(cur_image_vector)    #进行均值归一化
        last_index += cur_image_num
        histogram_code_space.extend(cur_image_vector)
    histogram_code_space = np.asarray(histogram_code_space)
    csvWrite('../result/coding/histogram_code_space_frequency'+str(n_cluster)+'.csv',histogram_code_space)
    print('写入histogram编码完成')
    print('特征空间维度：%d样本数 %d特征数'%np.shape(histogram_code_space))
    return histogram_code_space


# 获得SURF特征的簇分类标签
# 输入：Kmeans聚类模型:Kmeansmodel, 数据存储位置:path, 数据名:SURF_threshold
# 输出：SURF特征对应的簇分类标签:surfcluster
def getSURFCluster(Kmeansmodel, path, SURF_threshold):
    # 初始化生成器
    sampledata_float = dataGenerator(data_path=path, SURF_threshold=SURF_threshold, echos=1,
                                     minibatch_size=10000)
    surfcluster = []
    for cur_sample in sampledata_float:
        cur_label = Kmeansmodel.predict(cur_sample)
        surfcluster.extend(cur_label)
    print('SURF特征簇分类标签提取完成')
    return surfcluster


# 编码层实现
def histegramCoding(path, n_cluster, SURF_threshold=1000):
    start_time = time.time()
    Kmeansmodel = joblib.load(path + 'kmodel' + str(n_cluster) + 'SURF_threshold' + str(SURF_threshold) + '.pkl')
    surfcluster = getSURFCluster(Kmeansmodel, path, SURF_threshold)  # 获得SURF特征的簇分类标签

    #获得每幅图所含有的特征数量和对应标签
    f1 = open(path+'SURF_threshhold'+str(SURF_threshold)+'label_train.csv','r')
    csv_reader = csv.reader(f1)
    image_featurenum = []  #获得图片对应的特征数
    image_featurelabel = [] #获得图片对应的标签
    for line in csv_reader:
        image_featurenum.append(line[0])
        image_featurelabel.append(line[1])
    f1.close()

    # 获得wholelabel
    try:
        f2 = open('../result/coding/wholelabel.csv','r')
        csv_reader1 = csv.reader(f2)
        wholelabel=[]
        for line1 in csv_reader1:
            wholelabel.append(line1[1])
        f2.close()
    except:
        wholelabel = getWholelabel(image_featurelabel)
    finally:
        print('wholelabel读取成功')

    histegramcode = getHistogramCode(SURF_cluster=surfcluster,image_feature=image_featurenum,n_cluster=n_cluster)
    print(np.shape(histegramcode))
    histegramlabel = getHistogramlabel(image_featurelabel,wholelabel)
    end_time = time.time()

    f3 = open('../result/coding/Codingrecord.txt','a+')
    f3.write('写入的对应文件名字为: '+'histogram_code_space_frequency'+str(n_cluster)+'.csv'+'\n')
    f3.write('特征空间维度：%d样本数 %d特征数 %d标签数 \n'%(np.shape(histegramcode)[0],np.shape(histegramcode)[1],len(wholelabel)))
    f3.write('耗时为:%.8s \r\n'%(end_time-start_time))
    f3.close()

    return histegramcode,histegramlabel


if __name__ == '__main__':
    path = '../result/'
    SURF_threshold = 1000
    for n_cluster in range(80,201,40):
        print(n_cluster)
        histegramCoding(path,n_cluster)

