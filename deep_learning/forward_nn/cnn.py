from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from utils.utils import train_data_split


def cnn_1d(train):
    x_t, x_v, y_t, y_v = train_data_split(train)
    model = Sequential()
    model.add(Convolution1D(nb_filter=5, filter_length=1, input_shape=(1, 33)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_class))
    model.add(Activation('sigmoid'))

    model.predict()


def cnn_1d_(train):
    import numpy as np
    import keras

    x_t, x_v, y_t, y_v = train_data_split(train)
    data_1d = np.expand_dims(x_t, 0)
    print(data_1d)
    data_1d = np.expand_dims(data_1d, 2)

    # 定义卷积层
    filters = 1  # 卷积核数量为 1
    kernel_size = 5  # 卷积核大小为 5
    convolution_1d_layer = keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid',
                                                             input_shape=(x_t.shape[1], 1), activation="relu",
                                                             name="convolution_1d_layer")
    # 定义最大化池化层
    max_pooling_layer = keras.layers.MaxPool1D(pool_size=5, strides=1, padding="valid", name="max_pooling_layer")

    # 平铺层，调整维度适应全链接层
    reshape_layer = keras.layers.core.Flatten(name="reshape_layer")

    # 定义全链接层
    full_connect_layer = keras.layers.Dense(5, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1,
                                                                                                  seed=seed),
                                            bias_initializer="random_normal", use_bias=True, name="full_connect_layer")

    # 编译模型
    model = keras.Sequential()
    model.add(convolution_1d_layer)
    model.add(max_pooling_layer)
    model.add(reshape_layer)
    model.add(full_connect_layer)

    # 打印 full_connect_layer 层的输出
    output = keras.Model(inputs=model.input, outputs=model.get_layer('full_connect_layer').output).predict(data_1d)
    print(output)

    # 打印网络结构
    print(model.summary())

    return model

