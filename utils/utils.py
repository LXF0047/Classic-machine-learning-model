#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# 数据集划分
def train_data_split(df, label_name='label', id_name=None, _size=0.3):
    if label_name not in df.columns:
        print('[ERROR] The name of LABEL column is wrong')
        return None
    train_label = df[label_name]
    if id_name is not None:
        train_data = df.drop([id_name, label_name], axis=1)
    else:
        train_data = df.drop([label_name], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=_size, random_state=2020,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


# 根据字典数据画柱状图，多用于特征重要程度图
def draw_from_dict(dicdata, RANGE=None, axis=0):
    if RANGE is None:
        RANGE = len(dicdata)
    # dicdata：字典的数据。
    # RANGE：截取显示的字典的长度。
    # axis=0，代表条状图的柱子是竖直向上的。axis=1，代表柱子是横向的。考虑到文字是从左到右的，让柱子横向排列更容易观察坐标轴。
    by_value = sorted(dicdata.items(), key=lambda item: item[1], reverse=False)
    x = []
    y = []
    for d in by_value:
        x.append(d[0])
        y.append(d[1])
    if axis == 0:
        plt.bar(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    elif axis == 1:
        plt.barh(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    else:
        return "Axis got wrong value!"


def draw_confusion_matrix(matrix):
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax)  # 画热力图 , annot=True
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    # ax.autofmt_xdate()
    plt.show()


def draw_tsne(data, ncategories=2):
    xtsne = TSNE(n_components=ncategories, init='pca', random_state=2020)  # perplexity=30
    label = data['label']
    data.drop(['label'], axis=1, inplace=True)
    results = xtsne.fit_transform(data)
    vis_x = results[:, 0]
    vis_y = results[:, 1]
    plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("jet", ncategories))
    # plt.colorbar(ticks=range(2))
    # plt.clim(0.5, 2)
    plt.show()


def draw_roc(real, predict):
    fpr, tpr, thresholds = metrics.roc_curve(real, predict)
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    # 画ROC曲线
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.show()


# 检查字符转中是否有特殊字符
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def second2hms(dur):
    '''
    :param dur: 程序运行秒数
    :return: 时分秒
    '''
    return int(dur//3600), int((dur % 3600)//60), int(dur % 60)


if __name__ == '__main__':
    conn_xgb = [[55901, 225], [225, 15055]]
    conn_lgb = [[56119, 7], [12, 15268]]
    conn_rf = [[56119, 7], [18, 15262]]
    conn_lr = [[52286, 3840], [7630, 7650]]

    ssl_xgb = [[5032, 0], [0, 2428]]
    ssl_lgb = [[5029, 3], [2, 2426]]
    # ssl_rf = [[]]
    draw_confusion_matrix(ssl_xgb)