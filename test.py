from sklearn.datasets import load_iris
from machine_learning.ensemble_learning.boosting import BoostingModules
from deep_learning.forward_nn import ann
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils.utils import train_data_split
import xgboost as _xgb
from sklearn.metrics import f1_score


def load_data():
    # 读取莺尾花数据集
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.DataFrame(iris.target, columns=['target'])
    data_set = pd.concat([data, target], axis=1)
    return data_set


def load_eta_data():
    # (64554, 34)
    file_path = 'E:/竞赛数据/ETA/MTA-KDD-19-master/'

    d = pd.read_csv(file_path + 'kdd_19_data.csv')
    # eda(d, '')
    X_t, X_v, y_t, y_v = train_data_split(d, _size=0.2)
    X_t['label'] = y_t
    X_v['label'] = y_v
    X_t['id'] = X_t.index
    X_t['id'] = X_t.index
    # X_t.to_csv(file_path + 'eta_train.csv')
    # X_v.to_csv(file_path + 'eta_test.csv')
    return X_t, X_t


def train_handel(data):
    label = data['label']
    feature = data.drop(['id', 'label'], axis=1)
    return feature, label


def analysis(train):
    boost = BoostingModules(train)
    boost.rounds = 1000
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

    ng_model = boost.ng_model()

    # ANN
    X_t, X_v, y_t, y_v = train_data_split(train)
    ann_m = ann.Ann()
    ann_clf = ann_m.mlp(X_t, y_t)

    return xgb_model, lgb_model, cb_model, ann_clf, ng_model  # , ann_clf, ng_model


def cross_val(dt, test_df, k=3):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
    _result = 0  # 最终结果
    data_k, label_k = train_handel(dt)
    for train_index, test_index in kf.split(data_k, label_k):
        X_train_p, y_train_p = data_k.iloc[train_index], label_k.iloc[train_index]
        X_valid_p, y_valid_p = data_k.iloc[test_index], label_k.iloc[test_index]
        xgb_params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0
        }
        xgb_m = BoostingModules(train)
        xgb_m.X_t, xgb_m.X_v, xgb_m.y_t, xgb_m.y_v = X_train_p, X_valid_p, y_train_p, y_valid_p
        xgb_model = xgb_m.xgb_model(xgb_params)
        _result += xgb_model.predict(_xgb.DMatrix(test_df)) / k
    return [round(x) for x in _result]


def fit_predict(r):
    # fit
    xgb_c, lgb_c, cb_c, ann_c, ng_model = analysis(train, r)  # , ann_c, ng_model

    # predict
    test_xgb_res = xgb_c.predict(_xgb.DMatrix(test_feature))
    test_lgb_res = lgb_c.predict(test_feature)
    test_cb_res = cb_c.predict(test_feature)
    test_ann_res = ann_c.predict(test_feature)
    test_ngb_res = ng_model.predict(test_feature)

    # score
    test_xgb_res = [round(x) for x in test_xgb_res]
    test_lgb_res = [round(x) for x in test_lgb_res]
    test_ann_res = [round(x[0]) for x in test_ann_res.tolist()]

    xgb_f1_score = f1_score(test_xgb_res, test_label)  # 3
    lgb_f1_score = f1_score(test_lgb_res, test_label)  # 33
    cb_f1_score = f1_score(test_cb_res, test_label)  # 245
    ann_f1_score = f1_score(test_ann_res, test_label)  #
    ng_f1_score = f1_score(test_ngb_res, test_label)

    print('XGB: %s\nLGB: %s\nCB: %s\nANN: %s\nNGB: %s\n' % (
    xgb_f1_score, lgb_f1_score, cb_f1_score, ann_f1_score, ng_f1_score))


if __name__ == '__main__':
    # load data
    train, test = load_eta_data()
    test_feature, test_label = train_handel(test)
    fit_predict()