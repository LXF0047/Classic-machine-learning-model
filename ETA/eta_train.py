import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from machine_learning.ensemble_learning.boosting import BoostingModules
from utils.score import _f1_score


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


def train_model():
    # 加载数据
    train, test = load_data()
    boosting_model = BoostingModules(train)
    boosting_model.early_stop = 20
    boosting_model.rounds = 5000
    xgb_params = {
        'eta': 0.1,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'tree_method': 'gpu_hist'
    }
    xgb_m = boosting_model.xgb_model(xgb_params)

    # test
    test_label = test['label']
    test.drop(['label'], axis=1, inplace=True)
    pre = xgb_m.predict(xgb.DMatrix(test))
    pre = [round(x) for x in pre]
    score = _f1_score(pre, test_label)
    print('测试集F1值：%s' % score)


if __name__ == '__main__':
    train_model()