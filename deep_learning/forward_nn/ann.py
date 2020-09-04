import numpy as np
from utils.math import *
from keras.models import Sequential
from keras.layers import Dense


# 建立ANN
class NeuralNetwork:
    # 构造函数 layers指的是每层内有多少个神经元 layers内的数量表示有几层
    # acvitation 为使用的激活函数名称 有默认值 tanh 表示使用tanh(x)
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistics
            self.activation_deriv = logistics_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation = tanh_deriv

        self.weight = []
        # len(layers)-1的目的是 输出层不需要赋予相应的权值
        for i in range(1, len(layers) - 1):
            # 第一句是对当前层与前一层之间的连线进行权重赋值，范围在 -0.25 ~ 0.25之间
            self.weight.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            # 第二句是对当前层与下一层之间的连线进行权重赋值，范围在 -0.25 ~ 0.25之间
            self.weight.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # self是指引当前类的指针 X表示训练集 通常模拟成一个二维矩阵，每一行代表一个样本的不同特征
        # 每一列代表不同的样本  y指的是classLabel 表示的是输出的分类标记
        # learning_rate是学习率，epochs表示循环的次数
        X = np.atleast_2d(X)
        # 将X转换为numpy2维数组 至少是2维的
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        # X.shape[0]返回的是X的行数 X.shape[1]返回的是X的列数
        temp[:, 0:-1] = X  # :指的是所有的行 0:-1指的是从第一列到除了最后一列
        X = temp  # 偏向的赋值
        y = np.array(y)  # 将y转换为numpy array的形式

        # 使用抽样的算法 每次随机选一个 x中的样本
        for k in range(epochs):
            # randint(X.shape[0])指的是从0~X.shape[0] 之间随机生成一个int型的数字
            i = np.random.randint(X.shape[0])
            a = [X[i]]  # a是从x中任意抽取的一行数据

            # 正向更新
            for l in range(len(self.weight)):  # 循环遍历每一层
                # dot是求内积的运算 将内积运算的结果放在非线性转换方程之中
                a.append(self.activation(np.dot(a[l], self.weight[l])))

            error = y[i] - a[-1]  # 求误差 a[-1]指的是最后一层的classLabel
            deltas = [error * self.activation_deriv(a[-1])]

            # 开始反向传播 从最后一层开始，到第0层，每次回退1层
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weight[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()

            for i in range(len(self.weight)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])  # delta存的是误差
                self.weight[i] += learning_rate * layer.T.dot(delta)  # 误差与单元格的值的内积

    # 预测过程
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weight)):
            a = self.activation(np.dot(a, self.weight[l]))
        return a


class Ann:
    def __init__(self):
        self.seed = 2020

    def mlp(self, x, y):
        x = x.values
        y = y.values
        model = Sequential()
        model.add(Dense(68, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(68, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x=x, y=y, epochs=10, batch_size=100, validation_split=0.2)

        return model