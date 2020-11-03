#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import xgboost as xgb


def train_data_split(df, label_name='label', id_name=None, _size=0.3):
    from sklearn.model_selection import train_test_split
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


def load_black_data():
    # 共482793条
    file_path = '/home/lxf/data/DGA/word_dga/feature/'
    file_names = ['gozi', 'suppobox', 'matsnu']
    data = pd.read_csv(file_path + file_names[0])
    for file in file_names[1:]:
        tmp = pd.read_csv(file_path + file)
        data = pd.concat([data, tmp])
    data.drop(['family'], axis=1, inplace=True)
    print('DGA黑样本%s条读取完成' % data.shape[0])
    return data


def load_white_data(n=1667269):
    # 走到过滤分支的白样本
    data = pd.read_csv('/home/lxf/data/DGA/word_dga/feature/legit_words', nrows=n)
    data.drop(['family'], axis=1, inplace=True)
    print('DGA白样本%s条读取完成' % data.shape[0])
    return data


def load_white_others():
    # 没被过滤的走原来分支的白样本
    data = pd.read_csv('/home/lxf/data/DGA/word_dga/feature/legit_notwords')  # 437261
    data.drop(['family'], axis=1, inplace=True)
    print('DGA其他白样本%s条读取完成' % data.shape[0])
    return data


def xgb_model(X_t, X_v, y_t, y_v, t='train'):
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
    if t == 'train':
        num_rounds = 3709
        watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
        model = xgb.train(plst, xgb_train, num_rounds, watchlist)  # , early_stopping_rounds=20
        return model
    else:
        cv_res = xgb.cv(plst, xgb_train, 10000, nfold=5, early_stopping_rounds=30,
                        seed=2020, verbose_eval=2)
        print('XGB Best trees: %s' % cv_res.shape[0])
        return cv_res.shape[0]


def merg_train_data(sample_percent):
    black = load_black_data()
    black['label'] = 1
    # white = load_white_data(n=sample_percent*black.shape[0])
    white1 = load_white_data()
    white2 = load_white_others()
    white_all = pd.concat([white1, white2], ignore_index=True)
    white = white_all.reindex(np.random.permutation(white_all.index)).head(sample_percent * black.shape[0])
    white['label'] = 0
    print('[训练集] 黑样本数量：%s，白样本数量：%s' % (black.shape[0], white.shape[0]))
    train_data = pd.concat([black, white])
    train_data.to_csv('/home/lxf/data/DGA/word_dga/tmp/train.csv', index=False)
    return train_data


def train(model_name, sample_percent):
    '''
    :param model_name: 保存的模型文件名
    :param sample_percent: 采样时白样本是黑样本的几倍，最大7倍
    :return:
    '''
    # black 235785
    # white 1667269+437261=2104530
    # 加载训练集
    print('保存模型名称: %s, 黑白样本采样比例(黑 : 白) : 1:%s' % (model_name, sample_percent))
    train_data = merg_train_data(sample_percent)
    print(train_data['label'].value_counts())
    X_t, X_v, y_t, y_v = train_data_split(train_data)
    # 交叉验证
    # xgb_m = xgb_model(X_t, X_v, y_t, y_v, t='cv')
    # 训练
    xgb_m = xgb_model(X_t, X_v, y_t, y_v, t='train')
    # 保存模型
    xgb_m.save_model('/home/lxf/data/DGA/word_dga/modules/xgb_%s.model' % model_name)


def verification(model_name):
    # 验证模型效果
    xgb_m = xgb.Booster(model_file='/home/lxf/data/DGA/word_dga/modules/xgb_%s.model' % model_name)
    # lgb_m = lgb.Booster(model_file='/home/lxf/data/DGA/word_dga/modules/lgb_%s.model' % model_name)
    black_ = load_black_data()
    # 预测所有黑样本
    xgb_b_r = [round(x) for x in xgb_m.predict(xgb.DMatrix(black_))]
    # lgb_b_r = [round(x) for x in lgb_m.predict(black_)]
    xgb_black = (sum(xgb_b_r) / len(xgb_b_r))
    # lgb_black = (sum(lgb_b_r) / len(lgb_b_r))
    # 预测所有白样本
    white_ = load_white_data(n=1667268)
    white_others = load_white_others()
    white_all = pd.concat([white_, white_others])
    xgb_w_r = [round(x) for x in xgb_m.predict(xgb.DMatrix(white_all))]
    # lgb_w_r = [round(x) for x in lgb_m.predict(white_)]
    xgb_white_ = (1 - sum(xgb_w_r) / len(xgb_w_r))
    # lgb_white_ = (1 - sum(lgb_w_r) / len(lgb_w_r))

    # print('[黑样本准确率] XGB: %s, LGB: %s' % (xgb_black, lgb_black))
    # print('[白样本准确率] XGB: %s, LGB: %s' % (xgb_white_, lgb_white_))
    print('[黑样本准确率] XGB: %s' % xgb_black)
    print('[白样本准确率] XGB: %s' % xgb_white_)


def load_mul_data():
    file_path = '/home/lxf/data/DGA/word_dga/feature/'
    file_names = ['gozi', 'suppobox', 'matsnu']
    data = pd.read_csv(file_path + file_names[0])
    data['label'] = file_names[0]
    for file in file_names[1:]:
        tmp = pd.read_csv(file_path + file)
        tmp['label'] = file
        data = pd.concat([data, tmp])
    data.drop(['family'], axis=1, inplace=True)
    # data.to_csv('/home/lxf/data/DGA/word_dga/tmp/mul_train.csv', index=False)
    print(data['label'].value_counts())
    print('多分类训练数据加载完成')
    return data


def mul_train():
    from sklearn.metrics import f1_score
    data = load_mul_data()
    gozi = data[data['label'] == 'gozi'].head(116714)
    suppobox = data[data['label'] == 'suppobox'].head(116714)
    matsnu = data[data['label'] == 'matsnu']
    data2 = pd.concat([gozi, suppobox])
    data2 = pd.concat([data2, matsnu])

    category = {'gozi': 0, 'suppobox': 1, 'matsnu': 2}
    data2['label'] = data2['label'].map(lambda x: category[x])
    X_t, X_v, y_t, y_v = train_data_split(data2)
    xgb_m = xgb_mul_model(X_t, X_v, y_t, y_v)

    label = data2['label']
    data2.drop('label', axis=1, inplace=True)
    pre = xgb_m.predict(xgb.DMatrix(data2))

    score_micro = f1_score(label, pre, average='micro')
    score_macro = f1_score(label, pre, average='macro')
    print('F1 micro', score_micro)
    print('F1 macro', score_macro)


def xgb_mul_model(X_t, X_v, y_t, y_v):
    xgb_val = xgb.DMatrix(X_v, label=y_v)
    xgb_train = xgb.DMatrix(X_t, label=y_t)
    n_family = len(set(y_t.tolist()))
    xgb_params = {
        'eta': 0.1,
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': n_family,
        'eval_metric': 'mlogloss',
        'verbosity': 2,
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    }
    plst = list(xgb_params.items())
    num_rounds = 1846
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_rounds, watchlist)
    print(model.get_fscore())
    return model

    # cv_res = xgb.cv(plst, xgb_train, 10000, nfold=5, early_stopping_rounds=30,
    #                 seed=2020, verbose_eval=2)
    # print('XGB Best trees: %s' % cv_res.shape[0])


def main():
    # xgb_1p7.model是使用过滤后白样本的训练结果
    # xgb_1p7_2.model是使用所有白样本的训练结果
    # xgb_1p7_2_5000.model是使用所有白样本的训练结果(5k树)

    # for i in range(1, 8):
    #     model_name = '1p%s_2' % str(i)
    #     sample_percent = i
    #     train(model_name, sample_percent)
    #     verification(model_name)

    # 5000轮eta0.1实际树数量为4442
    model_name = '1p4_new'
    sample_percent = 4
    train(model_name, sample_percent)
    verification(model_name)


def test():
    from bin_retrain import extract_feature
    model_name = '1p7_2_5000_cp27'
    xgb_m = xgb.Booster(model_file='/home/lxf/data/DGA/word_dga/modules/xgb_%s.model' % model_name)
    res = ['bananan']
    # with open('/home/lxf/projects/tmp/check.txt', 'r') as r:
    #     for i in r:
    #         res.append(i.strip())
    #         break

    res_features = extract_feature(res)
    print(res_features)
    pre = xgb_m.predict(xgb.DMatrix(res_features))
    print('预测结果', pre)
    pre = [round(x) for x in pre]
    for index, item in enumerate(pre):
        if item == 1:
            print('恶意的索引', res[index])
    print(sum(pre), len(res))


if __name__ == '__main__':
    # main()  # 二分类训练
    # mul_train()  # 多分类训练
    # train('test', 1)
    # merg_train_data(3)
    mul_train()

    # ======结果记录======
    # 1:4 3709
    # [黑样本准确率]
    # XGB: 0.8339661096991878
    # [白样本准确率]
    # XGB: 0.9643212329219507

    # 1:3 3417
    # [黑样本准确率]
    # XGB: 0.9039961225618433
    # [白样本准确率]
    # XGB: 0.9466094313739559

    # 1:2 3259
    # [黑样本准确率]
    # XGB: 0.9470974102772824
    # [白样本准确率]
    # XGB: 0.9316231802935478

    # 1:1 3101
    # [黑样本准确率]
    # XGB: 0.9747614402031513
    # [白样本准确率]
    # XGB: 0.9058615965852692

    # 多分类 2289
    # micro  0.9961847002752733
    # macro  0.9960944878558157
    # 多分类训练集比例1:1:1  1846
    # micro    0.9954075774971297
    # macro    0.9954060912974617
