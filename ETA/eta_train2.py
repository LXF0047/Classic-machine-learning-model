import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def load_data():
    columns = pd.read_csv('/data1/zourzh/met/ai_columms.csv').columns.tolist()
    white_data = pd.read_csv('/data1/zourzh/met/white1.csv', header=None, iterator=True)  # , nrows=10000
    black_data = pd.read_csv('/data1/zourzh/met/black1.csv', header=None, iterator=True)  # , nrows=10000
    chunksize = 250000
    white_chunk = white_data.get_chunk(chunksize)
    black_chunk = black_data.get_chunk(chunksize)
    white_chunk.columns = columns
    black_chunk.columns = columns

    white_chunk['label'] = 0
    black_chunk['label'] = 1

    # 训练集使用数据集中前10w条数据，测试集中使用10w-25w条数据
    train = pd.concat([white_chunk.head(100000), black_chunk.head(100000)])
    test = pd.concat([white_chunk.tail(150000), black_chunk.tail(150000)])
    # 单独验证强特效果
    # feature_rank = {'flow_byte_dist16': 14, 'flow_byte_dist32': 14, 'dns_domain_alexaTop4': 6, 'dns_domain_alexaTop5': 7, 'flow_byte_dist2': 13, 'flow_byte_dist13': 6, 'flow_byte_dist35': 5, 'tls_server_ext9': 1, 'flow_byte_dist49': 2}
    # features = [x for x in feature_rank] + ['label']
    # train = train[features]
    # test = test[features]

    return train, test


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


def xgb_model(X_t, X_v, y_t, y_v):
    xgb_val = xgb.DMatrix(X_v, label=y_v)
    xgb_train = xgb.DMatrix(X_t, label=y_t)
    xgb_params = {
        'eta': 0.1,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'tree_method': 'gpu_hist'
    }
    plst = list(xgb_params.items())
    num_rounds = 5000
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_rounds, watchlist,
                      early_stopping_rounds=50)
    return model


def train_model():
    # 加载数据
    train, test = load_data()
    X_t, X_v, y_t, y_v = train_data_split(train)
    xgb_m = xgb_model(X_t, X_v, y_t, y_v)

    # test
    test_label = test['label']
    test.drop(['label'], axis=1, inplace=True)
    pre = xgb_m.predict(xgb.DMatrix(test))
    pre = [round(x) for x in pre]
    score = f1_score(test_label, pre, average='binary')
    print('测试集F1值：%s' % score)


if __name__ == '__main__':
    train_model()