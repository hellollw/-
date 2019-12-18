# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-04

"""
使用sklearn中的kmeans方法进行特征聚类

1. 使用轮廓系数作为聚类结果的评价指标（https://blog.csdn.net/weixin_44344462/article/details/89337770），轮廓系数范围为【-1，1】，越接近1说明聚类效果越好
2. 样本数据过大（内存分配不足）——使用partial_fit增量式学习方法（生成一个文件流迭代器）,同时使用miniBatchKmeans方法
3. 使用SSE的平均值来衡量聚类器的聚类效果
"""

import numpy as np
import joblib
import csv
from sklearn.cluster import MiniBatchKMeans
import time
from sklearn.metrics import silhouette_score


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


# 获得样本数据（使用生成器替换，数据维度过大！！）
# 输入：样本csv文件路径:path，需要读取的文件信息:SURF_threshold
# 输出：整理好的数据集:sampledata_float(无监督学习，只需要特征变量即可)
def getFeatureData(path, SURF_threshold):
    sampledata_str = []
    f1 = open(path + 'SURF_threshhold' + str(SURF_threshold) + 'des_train' + '.csv')

    csv_reader1 = csv.reader(f1)
    for i in csv_reader1:
        sampledata_str.append(i)  # 读出数据
    f1.close()
    sampledata_float = str2float(sampledata_str)
    print('数据转换完成')
    return sampledata_float


# 将样本数据放入Kmeans训练集中，保存Kmeans训练集
# 输入：样本csv文件路径:path，需要读取的文件信息:SURF_threshold, 簇中心数量:n_cluster，循环读取数量:echos， 生成器每次最小生成数量:mini_batch(默认5000）
# 输出：保存好的Kmeans训练集:Kmeans_model
def Kmeans_sk(path, SURF_threshold, n_cluster, echos, mini_batch=5000):
    """

    :param path: 样本csv文件路径
    :param SURF_threshold: 需要读取的文件信息
    :param n_cluster: 簇中心数量
    :param echos: 循环读取样本集的数目
    :return: 保存好的Kmeans训练集:Kmeans_model
    """
    start_time = time.time()
    sampledata_float = dataGenerator(data_path=path, SURF_threshold=SURF_threshold, echos=echos,
                                     minibatch_size=mini_batch)
    Kmeans_mode = MiniBatchKMeans(n_clusters=n_cluster, max_iter=500, batch_size=mini_batch, random_state=0)
    # batch_size为每次训练时抽取的样本容量
    for i, cur_sample_float in enumerate(sampledata_float):
        Kmeans_mode.partial_fit(cur_sample_float)
        # sparameter+=silhouette_score(cur_sample_float,Kmeans_mode.labels_)
        print('第%d轮完成' % i)
    # 使用SSE均值系数衡量分类器好坏
    SSE_mean = checkKmeansModel(Kmeans_mode,SURF_threshold)
    print('SSE系数:%f \r\n' % SSE_mean)
    # 计算平均轮廓系数（数据集过大，弃用）
    # s = silhouette_score(sampledata_float, Kmeans_mode.labels_)
    # s = Kmeans_mode.inertia_
    # 保存Kmeans模型
    joblib.dump(Kmeans_mode, path + 'kmodel' + str(n_cluster) + 'SURF_threshold' + str(SURF_threshold) + '.pkl')
    end_time = time.time()
    # 写入数据
    f = open(path + 'KmeansRecord.txt', 'a+')
    f.write('文件名: ' + path + 'kmodel' + str(n_cluster) + 'SURF_threshold' + str(SURF_threshold) + '.pkl' + '\n')
    f.write('Kmeans参数maxiter为:%d\n' % 500)
    f.write('读取的文件为:' + path + 'SURF_threshhold' + str(SURF_threshold) + 'des_train' + '.csv' + '\n')
    f.write('n_cluster:%d \n' % n_cluster)
    f.write('训练时间:%.8ss \n' % (end_time - start_time))
    f.write('SSE系数:%f \r\n' % SSE_mean)
    f.close()
    print('训练完成，耗时为:%.8s' % (end_time - start_time))


# 定义生成器，实现增量式学习，防止内存溢出的错误
# 输入:数据集位置:data_path, 需要读取的文件信息:SURF_threshold, 循环读取次数:echos,最小训练集数量:minibatch_size
# 返回：一个生成器
def dataGenerator(data_path, SURF_threshold, echos, minibatch_size):
    """
    定义生成器，实现增量式学习，防止内存溢出的错误
    :param data_path: 数据集位置
    :param minibatch_size: 最小训练集数量
    :return:返回一个生成器，一次初始化后相当于一个可迭代变量
    """
    X = []
    cur_line_num = 0

    for j in range(echos):
        f1 = open(data_path + 'SURF_threshhold' + str(SURF_threshold) + 'des_train' + '.csv', 'r')
        reader = csv.reader(f1)
        for line in reader:
            X.append(line)  # 这里要将数据转化成float类型
            cur_line_num += 1
            if cur_line_num >= minibatch_size:  # 当生成器当前生成的数据数量大于设定的最小数量大小时
                X_float = str2float(X)
                yield X_float
                X = []
                cur_line_num = 0
        f1.close()


# 设置检沿函数(检验模型的正确率）
# 输入：K聚类模型:Kmeans_model，需要读取的文件信息:SURF_threshold
# 输出：该模型的拟合程度（使用SSE系数作为指标）
def checkKmeansModel(Kmeans_model, SURF_threshold):
    # 读取Kmeans模型
    centers = Kmeans_model.cluster_centers_  # 读取簇中心
    SSE_mean = 0  # 初始化SSE均值
    i = 0
    # 生成器,默认为一轮迭代
    sampledata_float = dataGenerator(data_path=path, SURF_threshold=SURF_threshold, echos=1,
                                     minibatch_size=10000)
    for i, cur_sample in enumerate(sampledata_float):
        cur_label = Kmeans_model.predict(cur_sample)
        labels = []
        for x_label in cur_label:
            labels.append(centers[x_label, :])
        SSE_mean += calculateDist(np.mat(cur_sample), np.mat(labels))  # 计算平均SSE
    SSE_mean = SSE_mean / (i + 1)
    return SSE_mean


# 计算两个样本之间的欧氏距离作为评价标准（输入为矩阵)
# 输入：特征向量:x,簇中心:cluster_center
# 输出：平均欧氏距离dist
def calculateDist(x, cluster_center):
    m = np.shape(x)[0]
    dif = x - cluster_center
    dist = np.sum(np.multiply(dif,dif))  # 利用矩阵相乘
    return dist / m


if __name__ == '__main__':
    path = '../result/'
    SURF_threshold = 1000
    echos = 1
    n_cluster = 50
    Kmeans_sk(path, SURF_threshold, n_cluster,echos)
    # Kmeans_model1 = joblib.load(path + 'kmodel' + str(n_cluster) + 'SURF_threshold' + str(SURF_threshold) +'.pkl')
    # Kmeans_model2 = joblib.load(path + 'kmodel120' + 'SURF_threshold' + str(SURF_threshold) +'.pkl')
    # print(np.shape(Kmeans_model.cluster_centers_))
    # print(Kmeans_model.labels_)
    # print(len(Kmeans_model.labels_))
    # print('80:  ',checkKmeansModel(Kmeans_model1, SURF_threshold),'\r\n')
    # start = time.time()
    # print('80:  ',checkKmeansModel(Kmeans_model1,SURF_threshold))
    # end = time.time()
    # print('用时:%.8s'%(end-start))
