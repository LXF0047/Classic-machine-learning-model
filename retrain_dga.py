from machine_learning.ensemble_learning.boosting import BoostingModules
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
def load_data():
    dga_bin = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/dga_bin.csv'
    legit = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv'
    black = pd.read_csv(dga_bin)
    white = pd.read_csv(legit)
    print('black shape %s, %s' % (black.shape[0], black.shape[1]))
    print('white shape %s, %s' % (white.shape[0], white.shape[1]))

    # 合并黑白样本
    black['label'] = [1]*black.shape[0]
    white['label'] = [0]*white.shape[0]
    black['id'] = black.index
    white['id'] = white.index
    data = pd.concat([black, white])
    print(data.head())

    X_t, X_v, y_t, y_v = train_data_split(data, _size=0.2)
    X_t['label'] = y_t
    X_v['label'] = y_v
    X_t['id'] = X_t.index
    X_v['id'] = X_v.index
    return X_t, X_v


def analysis(train):
    boost = BoostingModules(train)
    boost.rounds = 5000
    boost.early_stop = 200
    xgb_params = {
                  'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'verbosity': 1
                  }
    xgb_model = boost.xgb_model(xgb_params)

    lgb_params = {
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': {'binary_logloss', 'auc'},
                 }
    lgb_model = boost.lgb_model(lgb_params)

    cb_model = boost.cb_model()

    # RF
    rf_model = random_forest.rf_model(train)

    return xgb_model, lgb_model, cb_model, rf_model


def fit_predict(train, test):
    test_feature, test_label = train_handel(test)
    # fit
    xgb_c, lgb_c, cb_c, rf_model = analysis(train)

    # predict
    test_xgb_res = xgb_c.predict(_xgb.DMatrix(test_feature))
    test_lgb_res = lgb_c.predict(test_feature)
    test_cb_res = cb_c.predict(test_feature)
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

    try:
        sns.heatmap(xgb_cm, annot=True, xticklabels=test.columns.tolist(), yticklabels=test.columns.tolist())
        plt.savefig('/home/lxf/figures/xgb_bin_cm.png')
        sns.heatmap(lgb_cm, annot=True, xticklabels=test.columns.tolist(), yticklabels=test.columns.tolist())
        plt.savefig('/home/lxf/figures/lgb_bin_cm.png')
        sns.heatmap(cb_cm, annot=True, xticklabels=test.columns.tolist(), yticklabels=test.columns.tolist())
        plt.savefig('/home/lxf/figures/cb_bin_cm.png')
        sns.heatmap(rf_cm, annot=True, xticklabels=test.columns.tolist(), yticklabels=test.columns.tolist())
        plt.savefig('/home/lxf/figures/rf_bin_cm.png')
        print('混淆矩阵图片保存成功')
    except:
        print('图片保存出错!')

    print('XGB: %s\nLGB: %s\nCB: %s\nRF: %s' % (xgb_f1_score, lgb_f1_score, cb_f1_score, rf_f1_score))


if __name__ == '__main__':
    train_set, test_set = load_data()
    fit_predict(train_set, test_set)
