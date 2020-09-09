from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from utils.utils import train_data_split
from keras.preprocessing import sequence
from keras.utils import to_categorical
from utils.score import _f1_score
import numpy as np



def lstm(train):
    x_t, x_v, y_t, y_v = train_data_split(train)
    x_train = np.reshape(x_t.values, (x_t.shape[0], x_t.shape[1], 1))
    x_test = np.reshape(x_v.values, (x_v.shape[0], x_v.shape[1], 1))
    y_train = to_categorical(y_t.values, num_classes=2)
    y_test = to_categorical(y_v.values, num_classes=2)

    # params
    nb_lstm_outputs = 30  # 神经元个数
    data_input = x_t.shape[1]

    # build model
    model = Sequential()
    model.add(LSTM(units=nb_lstm_outputs, input_shape=(data_input, 1)))
    model.add(Dense(2, activation='softmax'))

    # compile:loss, optimizer, metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train: epcoch, batch_size
    model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=1)

    model.summary()

    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('===LSTM F1 score %s===' % score)

    return model