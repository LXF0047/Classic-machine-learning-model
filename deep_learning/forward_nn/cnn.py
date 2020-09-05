from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from utils.utils import train_data_split
from utils.score import _f1_score


def cnn_1d_(train):
    import numpy as np
    import keras

    seed = 2020
    x_t, x_v, y_t, y_v = train_data_split(train)
    train = x_t.values.reshape(x_t.shape[0], x_t.shape[1], 1)
    label = y_t.values

    test = x_v.values.reshape(x_v.shape[0], x_v.shape[1], 1)
    test_label = y_v.values

    # 定义卷积层
    filters = 1  # 卷积核数量为 1
    kernel_size = 5  # 卷积核大小为 5
    convolution_1d_layer = keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='same',
                                                             input_shape=(x_t.shape[1], 1), activation="relu",
                                                             name="convolution_1d_layer")
    # 定义最大化池化层
    max_pooling_layer = keras.layers.MaxPool1D(pool_size=5, strides=1, padding="valid", name="max_pooling_layer")

    # 平铺层，调整维度适应全链接层
    reshape_layer = keras.layers.core.Flatten(name="reshape_layer")

    # 定义全链接层
    full_connect_layer = keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1,
                                                                                                  seed=seed),
                                            bias_initializer="random_normal", use_bias=True, activation='sigmoid',
                                            name="full_connect_layer")

    # 编译模型
    model = keras.Sequential()
    model.add(convolution_1d_layer)
    model.add(max_pooling_layer)
    model.add(reshape_layer)
    model.add(full_connect_layer)

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit
    model.fit(train, label, epochs=50, batch_size=100, validation_split=0.2)

    # 打印网络结构
    print(model.summary())

    # 验证集效果
    res = model.predict(test)
    _res = [round(x[0]) for x in res.tolist()]
    _score = _f1_score(_res, test_label)
    print('===F1 score: %s===' % _score)

    return model

