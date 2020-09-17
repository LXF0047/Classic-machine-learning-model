from machine_learning.ensemble_learning.boosting import BoostingModules
import time
from deep_learning.forward_nn import ann, cnn
from deep_learning.recurrent_nn import lstm
from keras.preprocessing import sequence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from utils.utils import train_data_split
import xgboost as _xgb
import numpy as np
from keras.utils import to_categorical
from utils.score import _f1_score, _confusion_metrix
from machine_learning.ensemble_learning import random_forest
from sklearn.model_selection import GridSearchCV
from utils.static_models import _xgboost
from multiprocessing import Process


# 去id和label列
def train_handel(data):
    label = data['label']
    feature = data.drop(['id', 'label'], axis=1)
    return feature, label


# 加载数据集
def load_bin_data(topn=100000, return_all=False):
    dga_bin = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/dga_bin.csv'
    legit = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv'
    black = pd.read_csv(dga_bin)
    white = pd.read_csv(legit)
    print('black shape %s, %s' % (black.shape[0], black.shape[1]))
    print('white shape %s, %s' % (white.shape[0], white.shape[1]))

    # 合并黑白样本
    if topn is None:
        topn_b, topn_w = black.shape[0], white.shape[0]
    else:
        topn_b, topn_w = topn, topn

    black['label'] = [1]*black.shape[0]
    white['label'] = [0]*white.shape[0]
    black['id'] = black.index
    white['id'] = white.index
    data = pd.concat([black.head(topn_b), white.head(topn_w)])
    data.drop(['type'], axis=1, inplace=True)

    X_t, X_v, y_t, y_v = train_data_split(data, _size=0.2)
    X_t['label'] = y_t
    X_v['label'] = y_v
    X_t['id'] = X_t.index
    X_v['id'] = X_v.index

    if return_all:
        return data.reindex(np.random.permutation(data.index))  # 打乱行顺序
    else:
        return X_t, X_v


def load_mul_data():
    dga_mul = '/data0/new_workspace/mlxtend_dga_multi_20190316/merge/new_feature/dga_multi.csv'
    mul_data = pd.read_csv(dga_mul)
    cols = ['type', 'domain_len', '_contains_digits', '_subdomain_lengths_mean', '_n_grams0', '_n_grams1',
            '_n_grams4', '_hex_part_ratio', '_alphabet_size', '_shannon_entropy', '_consecutive_consonant_ratio',
            'domain_seq35', 'domain_seq36', 'domain_seq38', 'domain_seq39', 'domain_seq40', 'domain_seq41',
            'domain_seq42', 'domain_seq43', 'domain_seq46', 'domain_seq47', 'domain_seq48', 'domain_seq49',
            'domain_seq50', 'domain_seq51', 'domain_seq52', 'domain_seq53', 'domain_seq54', 'domain_seq55',
            'domain_seq56', 'domain_seq57', 'domain_seq58', 'domain_seq59', 'domain_seq60', 'domain_seq61',
            'domain_seq62', 'domain_seq63', 'domain_seq64', 'domain_seq65', 'domain_seq66', 'domain_seq67',
            'domain_seq68', 'domain_seq69', 'domain_seq70', 'domain_seq71', 'domain_seq72', 'domain_seq73',
            'domain_seq74', 'domain_seq75']
    mul_data.columns = cols
    mul_data['id'] = mul_data.index
    mul_data['label'] = mul_data['type'].astype('int') - 1
    mul_data.drop(['type'], axis=1, inplace=True)
    X_t, X_v, y_t, y_v = train_data_split(mul_data, _size=0.9)
    X_t['label'] = y_t
    X_v['label'] = y_v
    X_t['id'] = X_t.index
    X_v['id'] = X_v.index
    return X_t, X_v


def bin_analysis(train):
    boost = BoostingModules(train)
    boost.rounds = 6000
    boost.early_stop = 200
    boost.modelname = 'bin'
    xgb_params = {
                  'eta': 0.1,
                  'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'verbosity': 1
                  }
    xgb_model = boost.xgb_model(xgb_params)

    lgb_params = {
                  'learning_rate': 0.1,
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': {'binary_logloss', 'auc'},
                 }
    lgb_model = boost.lgb_model(lgb_params)

    cb_model = boost.cb_model()

    # RF
    rf_model = random_forest.rf_model(train)

    return xgb_model, lgb_model, cb_model, rf_model


def mul_analysis(train):
    boost = BoostingModules(train)
    boost.rounds = 6000
    boost.early_stop = 200
    boost.modelname = 'mul'
    xgb_params = {
                  'booster': 'gbtree',
                  'objective': 'multi:softmax',
                  'num_class': 47,
                  'eval_metric': 'mlogloss',
                  'verbosity': 0
                  }
    xgb_model = boost.xgb_model(xgb_params)

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 47,
        'metric': 'multi_logloss',
    }
    lgb_model = boost.lgb_model(lgb_params)

    # from catboost import CatBoostClassifier
    # train['_contains_digits'] = train['_contains_digits'].astype('int')
    # train['_hex_part_ratio'] = train['_hex_part_ratio'].astype('int')
    # xt, xv, yt, yv = train_data_split(train)
    # # category_cols = ['_contains_digits', '_hex_part_ratio']
    # # category_id = []
    # # for index, value in enumerate(train.columns):
    # #     if value in category_cols:
    # #         category_id.append(index)
    # cb_model = CatBoostClassifier(iterations=6000, learning_rate=0.01, loss_function='MultiClass',
    #                            logging_level='Verbose', eval_metric='AUC')  # , cat_features=category_id
    # cb_model.fit(xt, yt, eval_set=(xv, yv), early_stopping_rounds=200)

    # return xgb_model, lgb_model, cb_model
    # return cb_model
    return xgb_model, lgb_model


def bin_fit_predict(train, test):
    test_feature, test_label = train_handel(test)
    # fit
    xgb_c, lgb_c, cb_c, rf_model = bin_analysis(train)

    # predict
    xgbstart = time.time()
    test_xgb_res = xgb_c.predict(_xgb.DMatrix(test_feature))
    xgbend = time.time()
    print('XGB预测用时：%s，速率：%s条/s' % (xgbend-xgbstart, round(test_feature.shape[0]/(xgbend-xgbstart))))
    time.sleep(2)
    lgbstart = time.time()
    test_lgb_res = lgb_c.predict(test_feature)
    lgbend = time.time()
    print('LGB预测用时：%s，速率：%s条/s' % (lgbend - lgbstart, round(test_feature.shape[0] / (lgbend - lgbstart))))
    time.sleep(2)
    cbstart = time.time()
    test_cb_res = cb_c.predict(test_feature)
    cbend = time.time()
    print('CB预测用时：%s，速率：%s条/s' % (cbend - cbstart, round(test_feature.shape[0] / (cbend - cbstart))))
    test_rf_res = rf_model.predict(test_feature)

    # score
    test_xgb_res = [round(x) for x in test_xgb_res]
    test_lgb_res = [round(x) for x in test_lgb_res]

    xgb_f1_score = _f1_score(test_xgb_res, test_label)
    lgb_f1_score = _f1_score(test_lgb_res, test_label)
    cb_f1_score = _f1_score(test_cb_res, test_label)
    rf_f1_score = _f1_score(test_rf_res, test_label)

    xgb_cm = _confusion_metrix(test_label, test_xgb_res)
    lgb_cm = _confusion_metrix(test_label, test_lgb_res)
    cb_cm = _confusion_metrix(test_label, test_cb_res)
    rf_cm = _confusion_metrix(test_label, test_rf_res)
    print('=' * 100)
    print(xgb_cm)
    print(lgb_cm)
    print(cb_cm)
    print(rf_cm)
    print('=' * 100)

    with open('/home/lxf/figures/cm.txt', 'w') as w:
        w.write('xgb\n' + str(xgb_cm) + '\n')
        w.write('lgb\n' + str(lgb_cm) + '\n')
        w.write('cb\n' + str(cb_cm) + '\n')
        w.write('rf\n' + str(rf_cm) + '\n')
        print('混淆矩阵保存成功')
    print('=' * 100)
    print('XGB: %s\nLGB: %s\nCB: %s\nRF: %s' % (xgb_f1_score, lgb_f1_score, cb_f1_score, rf_f1_score))


def mul_fit_predict(train, test):
    test_feature, test_label = train_handel(test)
    # fit
    xgb_c, lgb_c = mul_analysis(train)  # , cb_c
    # cb_c = mul_analysis(train)
    # predict

    test_xgb_res = xgb_c.predict(_xgb.DMatrix(test_feature))
    test_lgb_res = lgb_c.predict(test_feature)
    # test_cb_res = cb_c.predict(test_feature)
    # score
    mul_lgb_res = [list(x).index(max(x)) for x in test_lgb_res]

    xgb_f1_macro = _f1_score(test_xgb_res, test_label, avg='macro')
    lgb_f1_macro = _f1_score(mul_lgb_res, test_label, avg='macro')
    # cb_f1_macro = _f1_score(test_cb_res, test_label, avg='macro')
    xgb_f1_micro = _f1_score(test_xgb_res, test_label, avg='micro')
    lgb_f1_micro = _f1_score(mul_lgb_res, test_label, avg='micro')
    # cb_f1_micro = _f1_score(test_cb_res, test_label, avg='micro')

    xgb_cm = _confusion_metrix(test_label, test_xgb_res)
    lgb_cm = _confusion_metrix(test_label, mul_lgb_res)
    # cb_cm = _confusion_metrix(test_label, test_cb_res)

    print('=' * 100)
    print(xgb_cm)
    print(lgb_cm)
    # print(cb_cm)
    print('=' * 100)

    with open('/home/lxf/figures/cm_mul_xgb_lgb.txt', 'w') as w:
        w.write('xgb\n' + str(xgb_cm) + '\n')
        w.write('lgb\n' + str(lgb_cm) + '\n')
        # w.write('cb\n' + str(cb_cm) + '\n')
        print('多分类混淆矩阵保存成功')
    print('=' * 100)
    print('XGB macro %s, micro %s' % (xgb_f1_macro, xgb_f1_micro))
    print('LGB macro %s, micro %s' % (lgb_f1_macro, lgb_f1_micro))
    # print('CB macro %s, micro %s' % (cb_f1_macro, cb_f1_micro))


if __name__ == '__main__':
    # 二分类
    # train_set, test_set = load_bin_data()
    # print('训练数据数量%s, 测试数据数量%s' % (train_set.shape[0], test_set.shape[0]))
    # bin_fit_predict(train_set, test_set)
    # 多分类
    mul_train, mul_test = load_mul_data()
    print('训练数据数量%s, 测试数据数量%s' % (mul_train.shape[0], mul_test.shape[0]))
    mul_fit_predict(mul_train, mul_test)


    # 测性能
    # import lightgbm as lgb
    # from catboost import CatBoostClassifier
    # test_feature, test_label = train_handel(test_set)
    # # bin_xgb = _xgb.Booster(model_file='mul_xgb.model')
    # bin_lgb = lgb.Booster(model_file='mul_lgb.model')
    # # bin_cb = CatBoostClassifier().load_model('bin_cb.model')
    # print('start predict ...')
    # start = time.time()
    # for i in range(100):
    #     # xgb_res = bin_xgb.predict(_xgb.DMatrix(test_feature))
    #     res = bin_lgb.predict(test_feature)
    #     # res = cb_lgb.predict(test_feature)
    # end = time.time()
    # time_spend = end-start
    # sum_predict = test_feature.shape[0] * 100
    # print('%s条/s, %sms/条' % (round(sum_predict/time_spend), round(time_spend/sum_predict*1000, ndigits=5)))
