import numpy as np
import math


# 双曲函数
def tanh(x):
    return np.tanh(x)


# 双曲函数的微分
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


# 逻辑函数
def logistics(x):
    return 1 / (1 + np.exp(-x))


# 逻辑函数的微分
def logistics_derivative(x):
    return logistics(x) * (1 - logistics(x))


def safe_divide(a, b):
    if b == 0:
        b = 1
    return a / b


def max(array):
    """
    最大值
    :param array:   数据集
    :return:        数据集均值
    """
    if not array or len(array) < 1:
        return 0.0

    return float(np.max(array))


def mean(array):
    """
    均值
    :param array:   数据集
    :return:        数据集均值
    """
    if not array or len(array) < 1:
        return 0.0

    return float(np.mean(array))


def nth_var(array=(), nth_power=2):
    """
    n次方差
    :param array:       数据集
    :param nth_power:   2:方差, 3:3次方差, ...
    :return:            数据集n次方差
    """
    if not array or len(array) < 1:
        return 0.0

    average = mean(array)
    return sum(map(lambda x: pow(x-average, nth_power), array))/len(array)


def variance(array=()):
    """
    方差
    :param array:   数据集
    :return:        方差
    """
    return nth_var(array=array, nth_power=2)


def skweness(array=(), var=None):
    """
    偏度
    :param array:   数据集
    :param var:     数据集方差
    :return:        数据集偏度
    """
    if not array or len(array) < 1:
        return 0.0
    if var is None:
        var = variance(array=array)
    if var <= 0.0:
        return 0.0

    cubic_var = nth_var(array=array, nth_power=3)
    return cubic_var/pow(math.sqrt(var), 3)


def kurtosis(array=(), var=None):
    """
    峰度
    :param array:   数据集
    :param var:     数据集方差
    :return:        峰度
    """
    if not array or len(array) < 1:
        return 0.0
    if var is None:
        var = variance(array=array)
    if var <= 0.0:
        return 0.0

    fourth_var = nth_var(array=array, nth_power=4)
    return fourth_var/pow(var, 2) - 3


def entropy(prob):
    """
    熵
    :param prob: 概率
    :return:
    """
    return -1*prob*math.log(prob, 2)


def normalization(data):
    """
    归一化
    :param data:
    :return:
    """
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    """
    标准化
    :param data:
    :return:
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma