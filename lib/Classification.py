# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-05

"""
使用svm（one vs one）方法实现多分类（因图片样本经历编码过程后降维完成，所以不需要生成器）

1.送入svm的时候应该把字符串类型转化为string类型
"""
import numpy as np
import csv
from sklearn import svm, multiclass
import time


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


# 读取csv数据文件
def readCsv(type, datatype, n_cluster):
    """

    :param type:读取的数据种类（label or data)
    :param datatype:读取的数据集类型（train or test or wholelabel)
    :param n_cluster:特征数量
    :return: 数据集(list类型)data
    """

    path = ''
    if datatype == 'wholelabel':
        path = '../result/coding/wholelabel.csv'
    elif type == 'data':
        if datatype == 'train':
            path = '../result/coding/histogram_code_space_frequency' + str(n_cluster) + '.csv'
        elif datatype == 'test':
            path = '../result/coding/histogram_code_space_frequency_test' + str(n_cluster) + '.csv'
        else:
            raise NameError('输入格式错误')
    elif type == 'label':
        if datatype == 'train':
            path = '../result/SURF_threshhold1000label_train.csv'
        elif datatype == 'test':
            path = '../result/SURF_threshhold1000label_test.csv'
        else:
            raise NameError('输入格式错误')

    f = open(path, 'r')
    csv_reader = csv.reader(f)
    data = []
    for line in csv_reader:
        if type == 'label':  # 第二列为标签值
            data.append(line[1])
        elif type == 'data':
            data.append(line)  # 若为数据则加入整行
        else:
            raise NameError('只能输入label or data')
    f.close()
    return data


# 获得对应特征的标签值
def getHistogramlabel(image_feature, wholelabel):
    """

    :param image_feature:每一幅图片对应的字符串标签
    :param wholelabel: 编码的总标签
    :return: 编码空间的标签值
    """
    histogram_label = string2int(image_feature, wholelabel)
    print('获得样本标签完成')
    return histogram_label


def string2int(labelstring, wholelabel):
    labelint = []
    for cur_label in labelstring:
        label_index = wholelabel.index(cur_label)
        labelint.append(label_index)  # 转换为对应的种类数字
    return labelint


# 搭建one vs one的svm分类器，保存svm分类模型并且测试svm的准确率
def svmClassification(n_cluster, C):
    start_time = time.time()
    wholelabel = readCsv('label', 'wholelabel', n_cluster)

    train_data = np.asarray(readCsv('data', 'train', n_cluster))
    print(np.shape(train_data))
    train_label_str = readCsv('label', 'train', n_cluster)
    train_data = str2float(train_data)
    train_label_int = getHistogramlabel(train_label_str, wholelabel)
    print('训练集数据读取完成')
    test_data = np.asarray(readCsv('data', 'test', n_cluster))
    print(np.shape(test_data))
    test_data = str2float(test_data)
    test_label_str = readCsv('label', 'test', n_cluster)
    test_label_int = getHistogramlabel(test_label_str, wholelabel)
    print('测试集数据读取完成')

    svc_classifier = svm.SVC(C=C, kernel='rbf',gamma='scale')
    model = multiclass.OneVsOneClassifier(svc_classifier, -1)
    print('进入训练')
    clf = model.fit(train_data, train_label_int)
    print('训练完成')
    train_accuracy = clf.score(train_data, train_label_int)
    test_accuracy = clf.score(test_data, test_label_int)
    print('train_accuracy:%f' % train_accuracy)
    print('test_accuracy:%f' % test_accuracy)

    end_time = time.time()
    f = open('../result/svm/svmRecord.txt', 'a+')
    f.write('特征维度为:%d \n' % n_cluster)
    f.write('惩罚系数为:%d \n' % C)
    f.write('训练准确率: %.4f 测试准确率:%.4f\n' % (train_accuracy, test_accuracy))
    f.write('耗时:%.8s \r\n' % (end_time - start_time))
    f.close()

    print('耗时:%.8s' % (end_time - start_time))
    print('训练结束')


if __name__ == '__main__':
    for n_cluster in range(80,161,40):
        C = 10
        print(n_cluster)
        print(C)
        # C = 50
        svmClassification(n_cluster, C)
