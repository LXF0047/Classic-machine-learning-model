import numpy as np


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