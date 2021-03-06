#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
from keras.preprocessing import sequence
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import array as da
from xgboost.dask import DaskDMatrix
# from sklearn.metrics import f1_score
from utils.score import _f1_score, _confusion_metrix
from utils.utils import train_data_split
from utils import feature_extraction
from retrain_dga import load_bin_data, train_handel
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
from utils.utils import second2hms
import psutil


# 家族样本数量及多分类编号
lines_count = {'feodo': 192, 'randomloader': 5, 'symmi': 257816, 'volatile': 996, 'shifu': 2554, 'bebloh': 126527,
                   'oderoor': 1027, 'pykspa': 1440522, 'hesperbot': 192, 'proslikefan': 201131, 'matsnu': 116715,
                   'fobber': 200000, 'corebot': 246810, 'cryptowall': 94, 'pushdo': 203519, 'emotet': 321032,
                   'ekforward': 1460, 'ranbyus_v1': 15920, 'banjori': 439420, 'murofetweekly': 611920, 'rovnix': 437863,
                   'ccleaner': 11, 'dnschanger': 1599513, 'sphinx': 41621, 'tempedreve': 255, 'geodo': 384,
                   'beebone': 210, 'dnsbenchmark': 5, 'modpack': 52, 'pykspa_v2': 134648, 'bedep': 23180,
                   'tinynuke': 10176, 'chinad': 390080, 'padcrypt': 148800, 'tofsee': 920, 'szribi': 16007,
                   'vidro': 4900, 'torpig': 42120, 'sutra': 9882, 'vawtrak': 266982, 'gspy': 100, 'pandabanker': 9078,
                   'sisron': 10360, 'murofet': 7365890, 'virut': 5000000, 'xxhex': 4400, 'bamital': 133162,
                   'xshellghost': 11, 'omexo': 40, 'tsifiri': 59, 'darkshell': 49, 'tinba': 213607, 'mirai': 238,
                   'simda': 132233, 'gameoverp2p': 418000, 'ramnit': 132319, 'pizd': 2353, 'madmax': 181, 'ramdo': 104000,
                   'dircrypt': 57845, 'blackhole': 732, 'kraken': 133533, 'nymaim': 448743, 'gozi': 235786,
                   'ranbyus': 785648, 'unknownjs': 9630, 'redyms': 34, 'gameover': 5000000, 'qadars': 222088, 'dyre': 1381889,
                   'shiotob': 8003, 'bigviktor': 999, 'enviserv': 1306928, 'qakbot': 3170167, 'conficker': 1789506,
                   'necurs': 5992235, 'cryptolocker': 1786999, 'locky': 412003, 'suppobox': 130294}
family_label = {'feodo': 0, 'randomloader': 1, 'symmi': 2, 'volatile': 3, 'shifu': 4, 'bebloh': 5, 'oderoor': 6,
                    'pykspa': 7, 'hesperbot': 8, 'proslikefan': 9, 'matsnu': 10, 'fobber': 11, 'corebot': 12,
                    'cryptowall': 13, 'pushdo': 14, 'emotet': 15, 'ekforward': 16, 'ranbyus_v1': 17, 'banjori': 18,
                    'murofetweekly': 19, 'rovnix': 20, 'ccleaner': 21, 'dnschanger': 22, 'sphinx': 23, 'tempedreve': 24,
                    'geodo': 25, 'beebone': 26, 'dnsbenchmark': 27, 'modpack': 28, 'pykspa_v2': 29, 'bedep': 30,
                    'tinynuke': 31, 'chinad': 32, 'padcrypt': 33, 'tofsee': 34, 'szribi': 35, 'vidro': 36, 'torpig': 37,
                    'sutra': 38, 'vawtrak': 39, 'gspy': 40, 'pandabanker': 41, 'sisron': 42, 'murofet': 43, 'virut': 44,
                    'xxhex': 45, 'bamital': 46, 'xshellghost': 47, 'omexo': 48, 'tsifiri': 49, 'darkshell': 50,
                    'tinba': 51, 'mirai': 52, 'simda': 53, 'gameoverp2p': 54, 'ramnit': 55, 'pizd': 56, 'madmax': 57,
                    'ramdo': 58, 'dircrypt': 59, 'blackhole': 60, 'kraken': 61, 'nymaim': 62, 'gozi': 63, 'ranbyus': 64,
                    'unknownjs': 65, 'redyms': 66, 'gameover': 67, 'qadars': 68, 'dyre': 69, 'shiotob': 70, 'bigviktor': 71,
                    'enviserv': 72, 'qakbot': 73, 'conficker': 74, 'necurs': 75, 'cryptolocker': 76, 'locky': 77,
                    'suppobox': 78, 'others': 999}
# 邹哥采样的多分类数据样本家族对应关系
zrz_family_label = {'virut': 0, 'gameover': 1, 'murofet': 2, 'dnschanger': 3, 'dyre': 4, 'ranbyus': 5,
                    'murofetweekly': 6, 'banjori': 7, 'rovnix': 8, 'gameoverp2p': 9, 'chinad': 10, 'emotet': 11,
                    'vawtrak': 12, 'symmi': 13, 'corebot': 14, 'gozi': 15, 'qadars': 16, 'tinba': 17, 'pushdo': 18,
                    'padcrypt': 19, 'pykspa_v2': 20, 'kraken': 21, 'bamital': 22, 'simda': 23, 'bebloh': 24,
                    'matsnu': 25, 'ramdo': 26, 'dircrypt': 27, 'torpig': 28, 'sphinx': 29, 'bedep': 30, 'szribi': 31,
                    'sisron': 32, 'tinynuke': 33, 'sutra': 34, 'pandabanker': 35, 'vidro': 36, 'xxhex': 37, 'shifu': 38,
                    'pizd': 39, 'ekforward': 40, 'oderoor': 41, 'volatile': 42, 'blackhole': 43, 'geodo': 44,
                    'tempedreve': 45, 'mirai': 46}
# 去掉的特征
# c_drop = ['domain_seq35', 'domain_seq40', 'domain_seq36', 'domain_seq38', 'domain_seq39', 'domain_seq41',
#            'domain_seq42', '_hex_part_ratio', 'domain_seq43', 'domain_seq46', 'domain_seq47', '_n_grams4',
#            '_contains_digits', 'domain_seq48', 'domain_seq49', '_n_grams1', 'domain_seq50',
#            '_consecutive_consonant_ratio', 'domain_seq51', 'domain_len', 'domain_seq52', 'domain_seq53',
#            '_alphabet_size', 'domain_seq54', 'domain_seq75', 'domain_seq74', 'domain_seq72', 'domain_seq73',
#            'domain_seq55', '_subdomain_lengths_mean', 'domain_seq56', 'domain_seq57', 'domain_seq58',
#            'domain_seq59', '_shannon_entropy', '_n_grams0', 'domain_seq60', 'domain_seq61', 'domain_seq62',
#            'domain_seq63', 'domain_seq64', 'domain_seq65', 'domain_seq71', 'domain_seq67', 'domain_seq66',
#            'domain_seq68', 'domain_seq70', 'domain_seq69']
c_drop = ['domain_len', 'domain_seq73', 'domain_seq72', '_consecutive_consonant_ratio', 'domain_seq67', 'domain_seq66', '_alphabet_size', 'domain_seq48', 'domain_seq47', 'domain_seq46', 'domain_seq49', '_contains_digits', 'domain_seq65', 'domain_seq51', 'domain_seq59', 'domain_seq74', 'domain_seq56', 'domain_seq52', 'domain_seq53', 'domain_seq58', 'domain_seq60', 'domain_seq61', 'domain_seq40', 'domain_seq55', 'domain_seq75', '_subdomain_lengths_mean', 'domain_seq71', '_shannon_entropy', 'domain_seq70', 'domain_seq42', 'domain_seq63', 'domain_seq57', 'domain_seq69', 'domain_seq38', 'domain_seq64', 'domain_seq41', 'domain_seq62', 'domain_seq54', 'domain_seq43', 'domain_seq68', '_n_grams0', 'domain_seq50', '_n_grams1', '_hex_part_ratio', '_n_grams4', 'domain_seq36', 'domain_seq39', 'domain_seq35']

c_20000 = c_drop[15:]


def sampling_data(sampling=True, clf='bin', top=100000):
    '''
    数据集采样
    :param sampling: 是否采样，不采样时合并所有数据集
    :param clf: 数据集用于二分类还是多分类，标签列不同
    :param top: 每个样本随机取top个数据
    :return: 二分类返回带白样本的数据集，多分类返回带家族标签的数据集
    '''

    file_path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/csv_keras_new/'
    file_names = os.listdir(file_path)
    new_file_path = '/home/lxf/data/DGA/new_family/'   # 以前没有参加训练的样本
    new_file_names = os.listdir(new_file_path)

    # 白样本数量2440000

    # 各家族样本数量
    total_data = pd.read_csv(file_path + file_names[0])
    total_data['family'] = family_label[file_names[0]]  # 多分类数据添加家族名

    # 合并各家族数据
    for name in tqdm(file_names[1:] + new_file_names):
        if name in new_file_names:
            _path = new_file_path
        else:
            _path = file_path
        tmp_df = pd.read_csv(_path + name)
        tmp_df['family'] = family_label[name]
        if sampling:
            if lines_count[name] > top:
                tmp_df = tmp_df.reindex(np.random.permutation(tmp_df.index)).head(top)  # 打乱行序后取前十万条数据
                total_data = pd.concat([total_data, tmp_df])
            else:
                total_data = pd.concat([total_data, tmp_df])
        else:
            total_data = pd.concat([total_data, tmp_df])

    print('合并后数据集大小%s' % total_data.shape[0])
    print('不同家族取样数量:\n', total_data['family'].value_counts())

    if clf == 'bin':
        # 加载白样本
        legit_path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv'
        legit_data = pd.read_csv(legit_path)
        legit_data['type'] = 0

        total_data['type'] = 1
        total_data.drop(['family'], axis=1, inplace=True)  # 二分类去掉家族名列
        total_data = pd.concat([total_data, legit_data])
        return total_data
    else:
        return total_data


def generate_new_data():
    '''
    保存新采样数据
    :return: Null
    '''
    new_bin_data = sampling_data(sampling=True, clf='bin')
    new_bin_data.to_csv('/data1/lxf/DGA/data/bin_78_10w.csv', index=False)
    print('二分类数据写入完成')
    new_mul_data = sampling_data(sampling=True, clf='mul')
    new_mul_data.to_csv('/data1/lxf/DGA/data/mul_78_10w.csv', index=False)
    print('多分类数据写入完成')


def load_new_data(clf='bin', cols=None):
    '''
    加载新采样后数据
    :param clf: 加载二分类数据或多分类数据
    :param cols: 取部分列作为训练数据
    :return: 返回划分好的数据集 X_t, X_v, y_t, y_v
    '''
    if clf == 'bin':
        bin_data = pd.read_csv('/data1/lxf/DGA/data/bin_78_10w.csv')
        return train_data_split(bin_data, label_name='type')
    elif clf == 'mul':
        if cols is not None:
            new_id = [x for x in range(len(cols))]
            new_dict = dict(zip(cols, new_id))
            # 打印label和家族名称的对应顺序
            print(new_dict)
            mul_data = pd.read_csv('/data1/lxf/DGA/data/mul_78_10w.csv')
            mul_data = mul_data[mul_data['family'].isin(cols)]
            mul_data['family'] = mul_data['family'].apply(lambda x: new_dict[x])
        else:
            mul_data = pd.read_csv('/data1/lxf/DGA/data/mul_78_10w.csv')
        return train_data_split(mul_data, label_name='family')
    else:
        mul_test = pd.read_csv('/data1/lxf/DGA/data/mul_test.csv')
        label = mul_test['family']
        features = mul_test.drop(['family'], axis=1, inplace=True)
        return features, label


def load_zrz_data(clf='bin', min_features=False, int_type=False):
    '''
    :param clf: bin or mul
    :param min_features: 是否减少特征，减少特征的list为全局变量
    :param int_type: 是否将训练数据转化为int类型
    :return: 划分好的训练集和验证集
    '''
    if clf == 'mul':
        zrz_mul = '/data0/new_workspace/mlxtend_dga_multi_20190316/merge/new_feature/dga_multi.csv'
        mul_data = pd.read_csv(zrz_mul)
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
        mul_data['type'] = mul_data['type'].astype('int') - 1
        if min_features:
            mul_data.drop(c_20000, axis=1, inplace=True)
        if int_type:
            mul_data.astype('int')
        print('数据集读取完成, 特征数量：%s行， %s列' % (mul_data.shape[0], mul_data.shape[1]))
        return train_data_split(mul_data, label_name='type')
    else:
        dga_bin = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/dga_bin.csv'
        legit = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv'
        black = pd.read_csv(dga_bin)
        white = pd.read_csv(legit)
        black['label'] = [1] * black.shape[0]
        white['label'] = [0] * white.shape[0]
        data = pd.concat([black, white])
        data.drop(['type'], axis=1, inplace=True)
        print('数据集读取完成')
        return train_data_split(data)


def xgb_bin_model(X_t, X_v, y_t, y_v, c='train'):
    xgb_val = xgb.DMatrix(X_v, label=y_v)
    xgb_train = xgb.DMatrix(X_t, label=y_t)
    xgb_params = {
        'eta': 0.1,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 2,
        'tree_method': 'gpu_hist'
    }
    plst = list(xgb_params.items())
    num_rounds = 1200  # 迭代次数,cv10折交叉验证结果
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    if c == 'train':
        model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)# , feval=f1_score, maximize=True
        return model
    else:
        cv_res = xgb.cv(plst, xgb_train, num_rounds, nfold=5, early_stopping_rounds=100,
                        seed=2020, verbose_eval=2)
        print('XGB Best trees: %s' % cv_res.shape[0])
        return cv_res.shape[0]


def lgb_bin_model(X_t, X_v, y_t, y_v, c='train'):
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
    }
    lgb_train = lgb.Dataset(X_t, y_t)
    lgb_eval = lgb.Dataset(X_v, y_v, reference=lgb_train)
    if c == 'train':
        gbm = lgb.train(params, lgb_train, num_boost_round=1434, valid_sets=lgb_eval)
        # print(gbm.feature_importance())
        return gbm
    else:
        n_est = lgb.cv(params, lgb_train, num_boost_round=5000, nfold=5, early_stopping_rounds=100, seed=2020,
                       verbose_eval=2)
        print('LGB Best trees: %s' % len(n_est['auc-mean']))
        return n_est


def lgb_mul_model(X_t, X_v, y_t, y_v, c='train'):
    n_family = len(set(y_t.tolist()))
    lgb_train = lgb.Dataset(X_t, y_t)
    lgb_eval = lgb.Dataset(X_v, y_v, reference=lgb_train)
    lgb_params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': n_family,
        'metric': 'multi_logloss',
        'nthread': 56
        # 'device': 'gpu',
        # 'gpu_platform_id': 1,
        # 'gpu_device_id': 1,
    }
    if c == 'train':
        gbm = lgb.train(lgb_params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval,
                        early_stopping_rounds=50)
        return gbm
    else:
        n_est = lgb.cv(lgb_params, lgb_train, num_boost_round=2000, nfold=5, early_stopping_rounds=50, seed=2020,
                       verbose_eval=2)
        return n_est


def xgb_mul_model(X_t, X_v, y_t, y_v, c='train', n_tree=100):
    # print('===XGB多分类模型【训练集】各家族样本数量===')
    # print(y_t.value_counts())
    # print('=====================================')
    # print('===XGB多分类模型【测试集】各家族样本数量===')
    # print(y_v.value_counts())
    # print('=====================================')
    xgb_val = xgb.DMatrix(X_v, label=y_v)
    xgb_train = xgb.DMatrix(X_t, label=y_t)
    n_family = len(set(y_t.tolist()))
    xgb_params = {
        'eta': 0.3,
        'subsample': '0.7',
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': n_family,
        'eval_metric': 'mlogloss',
        'verbosity': 2,
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    }
    plst = list(xgb_params.items())
    num_rounds = n_tree  # 迭代次数
    if c == 'train':
        watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
        tmp = time.time()
        model = xgb.train(plst, xgb_train, num_rounds, watchlist)  #, early_stopping_rounds=50
        multigpu_time = time.time() - tmp
        h, m, s = second2hms(multigpu_time)
        print("Single GPU Training Time: %s h %s m %s s" % (h, m, s))
        # print(model.get_fscore())
        return model
    elif c == 'cv':
        cv_res = xgb.cv(plst, xgb_train, num_rounds, nfold=5, early_stopping_rounds=50,
                        maximize=True, seed=2020, verbose_eval=2)
        return cv_res.shape[0]


# xgb多gpu训练准备并行训练数据
def load_higgs_for_dask(client, X_t, X_v, y_t, y_v):
    '''
    :param client: gpu设备
    :param X_t: 训练集
    :param X_v: 验证集
    :param y_t: 训练集标签
    :param y_v: 验证集标签
    :return: dask.datafram格式的数据
    '''
    import dask.dataframe as dd
    # 1. Create a Dask Dataframe from Pandas Dataframe.
    ddf_higgs_train = dd.from_pandas(X_t, npartitions=8)
    ddf_higgs_test = dd.from_pandas(X_v, npartitions=8)
    ddf_y_train = dd.from_pandas(y_t, npartitions=8)
    ddf_y_test = dd.from_pandas(y_v, npartitions=8)
    # 2. Create Dask DMatrix Object using dask dataframes
    ddtrain = DaskDMatrix(client, ddf_higgs_train, ddf_y_train)
    ddtest = DaskDMatrix(client, ddf_higgs_test, ddf_y_test)

    return ddtrain, ddtest


# xgb多GPU并行做多分类预测
def xgb_mul_gpu_train(X_t, X_v, y_t, y_v):
    '''
    :param X_t: train
    :param X_v: test
    :param y_t: train label
    :param y_v: test label
    :return: fitted model
    '''
    # https://xgboost.readthedocs.io/en/latest/gpu/index.html  xgb官方文档
    # https://examples.dask.org/machine-learning/xgboost.html  dask官方文档
    # https://towardsdatascience.com/lightning-fast-xgboost-on-multiple-gpus-32710815c7c3  案例
    # https://gist.github.com/MLWhiz/44d39ab3a01fe4e57c974133276705f9  数据集并行处理方式
    # pip install fsspec>=0.3.3

    n_family = len(set(y_t.tolist()))

    with LocalCUDACluster(n_workers=8) as cluster:
        with Client(cluster) as client:
            print('数据集并行化处理')
            ddtrain, ddtest = load_higgs_for_dask(client, X_t, X_v, y_t, y_v)
            param = {'objective': 'multi:softmax',
                     'eta': 0.3,
                     'subsample': '0.7',
                     'num_class': n_family,
                     'eval_metric': 'mlogloss',
                     'verbosity': 2,
                     'tree_method': 'gpu_hist',
                     }
            # 'nthread': -1
            print("多GPU训练开始 ...")
            tmp = time.time()
            output = xgb.dask.train(client, param, ddtrain, num_boost_round=1000, evals=[(ddtest, 'test')])
            multigpu_time = time.time() - tmp
            print('训练完成')
            bst = output['booster']
            multigpu_res = output['history']
            h, m, s = second2hms(multigpu_time)
            print("Multi GPU Training Time: %s h %s m %s s" % (h, m, s))
    return bst


def bin_train():
    # 使用邹哥采样的训练数据，加载带id和label的二分类数据集，将数据集分为训练用和测试用
    # train_set, test_set = load_bin_data(topn=150000)
    # test_feature, test_label = train_handel(test_set)
    # X_t, X_v, y_t, y_v = train_data_split(train_set)

    # 使用所有数据的数据集
    X_t, X_v, y_t, y_v = load_new_data(clf='bin')

    # xgb_cv = xgb_bin_model(X_t, X_v, y_t, y_v, c='cv')  # 500
    # lgb_cv = lgb_bin_model(X_t, X_v, y_t, y_v, c='cv')  # 4

    xgb_ = xgb_bin_model(X_t, X_v, y_t, y_v, c='train')  # 500
    # lgb_ = lgb_bin_model(X_t, X_v, y_t, y_v, c='train')  # 4
    #
    # # 使用新采样的二分类数据
    xgb_.save_model('bin_xgb_1200trees.model')
    # lgb_.save_model('bin_lgb_1434trees.model')

    xgb_df = verification(clf='bin', model_csv_name='bin_xgb_1200trees')
    # lgb_df = verification(clf='bin', model_csv_name='bin_xgb_500trees')


def xgb_mul_train(c=None):
    print('XGB 多分类训练开始...')
    X_t, X_v, y_t, y_v = load_new_data(clf='mul', cols=c)
    # X_t, X_v, y_t, y_v = load_zrz_data(clf='mul')
    mul_clf = xgb_mul_model(X_t, X_v, y_t, y_v, c='train')
    mul_clf.save_model('mul_xgb_test1.model')
    print('模型保存成功')


def lgb_mul_train(c=None):
    print('LGB 多分类训练开始...')
    X_t, X_v, y_t, y_v = load_new_data(clf='mul', cols=c)
    # X_t, X_v, y_t, y_v = load_zrz_data(clf='mul')
    mul_clf = lgb_mul_model(X_t, X_v, y_t, y_v, c='train')
    mul_clf.save_model('mul_lgb_test1.model')
    print('模型保存成功')


def xgb_bin_gridsearch():
    '''

    :return:
    '''
    # 使用邹哥采样的训练数据，加载带id和label的二分类数据集，将数据集分为训练用和测试用
    train_set, test_set = load_bin_data(topn=400000)

    X_t, X_v, y_t, y_v = train_data_split(train_set)
    n_es = xgb_bin_model(X_t, X_v, y_t, y_v)

    param_test1 = {
        'max_depth': range(3, 8, 2),
        'min_child_weight': range(2, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=n_es, max_depth=6,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=-1, scale_pos_weight=1,
                                                    seed=2020),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=-1, cv=10, verbose=2)
    gsearch1.fit(X_t, y_t)
    print('训练使用数据量 ：%s' % train_set.shape[0])
    print('n_estimators = %s\nmax_depth = %s\nmin_child_weight = %s\nbest_score_ = %s' % (n_es,
            gsearch1.best_params_['max_depth'], gsearch1.best_params_['min_child_weight'], gsearch1.best_score_))


def xgb_mul_gridsearch():
    # 使用邹哥采样的训练数据
    X_t, X_v, y_t, y_v = load_zrz_data(clf='mul')
    # print('开始寻找最佳n_estimators')
    # n_es = xgb_mul_model(X_t, X_v, y_t, y_v)
    # print('最佳n_estimators：%s， 网格搜索参数开始' % n_es)
    n_es = 151
    param_test1 = {
        'max_depth': range(3, 8, 2),
        'min_child_weight': range(2, 6, 2)
    }
    # , tree_method = 'gpu_hist', gpu_id = 0
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=n_es, max_depth=6,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='multi:softmax', seed=2020),
                            param_grid=param_test1, scoring='roc_auc', cv=5, verbose=2, n_jobs=28)
    gsearch1.fit(X_t, y_t)
    print('n_estimators = %s\nmax_depth = %s\nmin_child_weight = %s\nbest_score_ = %s' % (n_es,
                                                                                          gsearch1.best_params_[
                                                                                              'max_depth'],
                                                                                          gsearch1.best_params_[
                                                                                              'min_child_weight'],
                                                                                          gsearch1.best_score_))


def lgb_bin_gridsearch():
    # 使用邹哥采样的训练数据，加载带id和label的二分类数据集，将数据集分为训练用和测试用
    train_set, test_set = load_bin_data(topn=400000)

    X_t, X_v, y_t, y_v = train_data_split(train_set)
    n_es = xgb_bin_model(X_t, X_v, y_t, y_v)

    param_test1 = {
        'max_depth': range(3, 8, 2),
        'min_child_weight': range(2, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=n_es, max_depth=6,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=-1, scale_pos_weight=1,
                                                    seed=2020),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=-1, cv=10, verbose=2)
    gsearch1.fit(X_t, y_t)
    print('训练使用数据量 ：%s' % train_set.shape[0])
    print('n_estimators = %s\nmax_depth = %s\nmin_child_weight = %s\nbest_score_ = %s' % (n_es,
                                                                                          gsearch1.best_params_[
                                                                                              'max_depth'],
                                                                                          gsearch1.best_params_[
                                                                                              'min_child_weight'],
                                                                                          gsearch1.best_score_))


def lgb_mul_gridsearch():
    # 使用邹哥采样的训练数据
    X_t, X_v, y_t, y_v = load_zrz_data(clf='mul')
    print('开始寻找最佳n_estimators')
    # n_es = lgb_mul_model(X_t, X_v, y_t, y_v, c='cv')
    n_es = 19

    parameters = {
        'max_depth': [10, 15, 20, 25],
        'learning_rate': [0.1, 0.15, 0.2],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_freq': [2, 4, 5, 6, 8],
        'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
        'lambda_l2': [0, 10, 15, 35, 40],
    }

    gbm = lgb.LGBMClassifier(objective='multiclass',
                             metric='multi_logloss',
                             learning_rate=0.1,
                             num_iterations=n_es,
                             n_jobs=32,
                             )
    gsearch = GridSearchCV(gbm, param_grid=parameters, cv=5, n_jobs=32, verbose=2)
    gsearch.fit(X_t, y_t)
    print('参数的最佳取值:{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))


def get_new_family():
    # 将360中的新家族写入到文件
    shiotob = []
    bigviktor = []
    enviserv = []
    with open('/home/lxf/projects/test/whitedga.txt', 'r') as r:
        for line in r:
            if line.startswith('shiotob'):
                shiotob.append(line.strip())
            elif line.startswith('bigviktor'):
                bigviktor.append(line.strip())
            elif line.strip('enviserv'):
                enviserv.append(line.strip())

    new2 = ['shiotob', 'bigviktor', 'enviserv']  # 新家族
    for name in new2:
        with open('/home/lxf/data/DGA/new_family_raw/%s.csv' % name, 'w') as w:
            if name == 'shiotob':
                for i in shiotob:
                    w.write(i.split('\t')[1] + '\n')
                print('shiotob写入完成, 样本数量：%s' % len(shiotob))
            elif name == 'bigviktor':
                for i in bigviktor:
                    w.write(i.split('\t')[1] + '\n')
                print('bigviktor写入完成, 样本数量：%s' % len(bigviktor))
            elif name == 'enviserv':
                for i in enviserv:
                    w.write(i.split('\t')[1] + '\n')
                print('enviserv写入完成, 样本数量：%s' % len(enviserv))


def extract_feature(domains):
    extractfeatures = feature_extraction.DGAExtractFeatures()
    features = extractfeatures.extract_all_features(domains)
    valid_chars = {'-': 1, '.': 2, '1': 3, '0': 4, '3': 5, '2': 6, '5': 7, '4': 8, '7': 9, '6': 10, '9': 11, '8': 12,
                   '_': 13, 'a': 14, 'c': 15, 'b': 16, 'e': 17, 'd': 18, 'g': 19, 'f': 20, 'i': 21, 'h': 22, 'k': 23,
                   'j': 24, 'm': 25, 'l': 26, 'o': 27, 'n': 28, 'q': 29, 'p': 30, 's': 31, 'r': 32, 'u': 33, 't': 34,
                   'w': 35, 'v': 36, 'y': 37, 'x': 38, 'z': 39}

    handle_domain_list = [[valid_chars[y] for y in x] for x in domains]
    domain_sequence = sequence.pad_sequences(handle_domain_list, maxlen=75)
    del_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                 27,
                 28, 29, 30, 31, 32, 33, 36, 43, 44]
    x1 = np.delete(domain_sequence, del_index, axis=1)
    features = np.hstack((features, x1))

    return features


def new2vectors():
    path2 = '/home/lxf/data/DGA/word_dga/raw/'
    new2 = ['legit_notwords']
    cols = ['domain_len', '_contains_digits', '_subdomain_lengths_mean', '_n_grams0', '_n_grams1',
            '_n_grams4', '_hex_part_ratio', '_alphabet_size', '_shannon_entropy', '_consecutive_consonant_ratio',
            'domain_seq35', 'domain_seq36', 'domain_seq38', 'domain_seq39', 'domain_seq40', 'domain_seq41',
            'domain_seq42', 'domain_seq43', 'domain_seq46', 'domain_seq47', 'domain_seq48', 'domain_seq49',
            'domain_seq50', 'domain_seq51', 'domain_seq52', 'domain_seq53', 'domain_seq54', 'domain_seq55',
            'domain_seq56', 'domain_seq57', 'domain_seq58', 'domain_seq59', 'domain_seq60', 'domain_seq61',
            'domain_seq62', 'domain_seq63', 'domain_seq64', 'domain_seq65', 'domain_seq66', 'domain_seq67',
            'domain_seq68', 'domain_seq69', 'domain_seq70', 'domain_seq71', 'domain_seq72', 'domain_seq73',
            'domain_seq74', 'domain_seq75']
    for name in new2:
        with open(path2 + '%s' % name, 'r') as r:
            res = extract_feature([x.strip() for x in r])  # [x.strip() for x in r]
            np.savetxt('/home/lxf/data/DGA/word_dga/feature/%s' % name, res, delimiter=',')

        tmp = pd.read_csv('/home/lxf/data/DGA/word_dga/feature/%s' % name)
        tmp.columns = cols
        tmp['family'] = name
        tmp.to_csv('/home/lxf/data/DGA/word_dga/feature/%s' % name, index=False)
        print('%s保存成功，数量：%s' % (name, tmp.shape[0]))


def verification(clf, zrz=False, c=None, model_csv_name=None, min_features=False, int_type=False):
    '''
    :param clf: mul or bin
    :param zrz: 是否采用现有采样数据
    :param c: 采用的家族，在zrz=False时有效
    :param model_csv_name: 加载模型和保存预测结果的名字，e.g. model_csv_name=test  >>>  test.model test.csv
    :param min_features: 是否减少特征，减少的特征为一全局list  （尝试提高训练速度）
    :param int_type: 是否将训练数据转为int类型  （尝试提高训练速度）
    :return: 测试结果的csv包括：家族名称，总数量，未命中数量，准确率，预测速度
    '''
    # 邹哥模型用到的家族
    if clf == 'mul':
        file_path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/csv_keras_new/'
        file_names = os.listdir(file_path)

        # 新加入的家族
        new_file_path = '/home/lxf/data/DGA/new_family/'  # 以前没有参加训练的样本
        new_file_names = os.listdir(new_file_path)
        # 白样本
        # legit_path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv'
        # legit_data = pd.read_csv(legit_path)
        # legit_data.drop(['type'], axis=1, inplace=True)
    else:
        file_path = ''
        new_file_path = ''
        file_names = []
        new_file_names = []
    # 加载训练好的模型
    if 'xgb' in model_csv_name:
        model = xgb.Booster(model_file='%s.model' % model_csv_name)  # 使用邹哥采样的数据重训练的xgb多分类模型，未进行调参
    elif 'lgb' in model_csv_name:
        model = lgb.Booster(model_file='%s.model' % model_csv_name)
    else:
        model = joblib.load('%s.pkl' % model_csv_name)

    # 进行预测
    if clf == 'mul':
        # pid = os.getpid()
        total_time = 0  # 预测总用时
        total_sample = 0  # 预测样本总条数

        # 写入结果csv中内容
        res_df = {'family': [], 'unhit': [], 'total': [], 'hit_ratio': [], 'speed': []}
        print('开始预测')
        if c is not None:
            _filenames = c  # + ['others']
        else:
            _filenames = new_file_names + file_names
        for name in _filenames:
            # 要预测的数据
            if zrz:
                notused = ['feodo', 'randomloader', 'pykspa', 'hesperbot', 'proslikefan', 'fobber', 'cryptowall',
                           'ranbyus_v1', 'ccleaner', 'beebone', 'dnsbenchmark', 'modpack', 'tofsee', 'gspy',
                           'xshellghost', 'omexo', 'tsifiri', 'darkshell', 'ramnit', 'madmax', 'nymaim',
                           'unknownjs', 'redyms']
                if name not in new_file_names + notused:
                    tmp = pd.read_csv(file_path + name)
                    if min_features:
                        tmp.drop(c_20000, axis=1, inplace=True)
                    if int_type:
                        tmp.astype('int')
                else:
                    continue
            else:
                if name in new_file_names:
                    tmp = pd.read_csv(new_file_path + name)
                    tmp.drop(['family'], axis=1, inplace=True)
                # elif name == 'others':
                #     tmp = pd.read_csv('/data1/lxf/DGA/data/mul_78_10w.csv')
                #     tmp.drop(['family'], axis=1, inplace=True)
                #     tmp = tmp.sample(n=100000, axis=0)
                else:
                    tmp = pd.read_csv(file_path + name)

            pre_start = time.time()  # 开始预测时间
            if 'xgb' in model_csv_name:
                res = model.predict(xgb.DMatrix(tmp))  # XGB预测结果
            elif 'lgb' in model_csv_name:
                lgb_res = model.predict(tmp)
                res = [list(x).index(max(x)) for x in lgb_res]  # LGB预测结果
            else:
                res = model.predict(tmp)

            pre_end = time.time()  # 结束预测时间
            pre_time = pre_end - pre_start  # 预测用时

            total_time += pre_time
            total_sample += tmp.shape[0]

            if c is not None:
                old_id = [family_label[x] for x in c]
                new_id = [x for x in range(len(old_id))]
                new_dict = dict(zip(old_id, new_id))  # 将选取列的id号重置为[0, num)
                false_count = len([x for x in res if x != new_dict[family_label[name]]])
            else:
                if zrz:
                    false_count = len([x for x in res if x != zrz_family_label[name]])
                else:
                    false_count = len([x for x in res if x != family_label[name]])

            print('家族：%s, 误报个数：%s, 正确率：%s, 速率： %s ms/条' % (name, false_count, round(1 - false_count/tmp.shape[0], 4), round(pre_time*1000/tmp.shape[0], 5)))
            # print('PID: %s , Used Memory: %s MB' % (pid, psutil.Process(pid).memory_info().rss / 1024 / 1024))

            # 存入结果csv
            res_df['family'].append(name)
            res_df['unhit'].append(false_count)
            res_df['total'].append(tmp.shape[0])
            res_df['hit_ratio'].append(round(1 - false_count/tmp.shape[0], 4))
            res_df['speed'].append(round(pre_time*1000/tmp.shape[0], 5))

        df2save = pd.DataFrame(res_df)
        # df2save.to_csv('/home/lxf/data/DGA/training_results/%s_res.csv' % model_csv_name, index=False)
        print('速率：%s 条/s，%s ms/条' % (round(total_sample/total_time, 5), round(total_time*1000/total_sample, 5)))
        return df2save

    else:
        if 'xgb' in model_csv_name:
            model = xgb.Booster(model_file='%s.model' % model_csv_name)
        elif 'lgb' in model_csv_name:
            model = lgb.Booster(model_file='%s.model' % model_csv_name)
        data = pd.read_csv('/data1/lxf/DGA/data/all_bin_data.csv')
        samples = data.shape[0]
        label = data['type']
        data.drop(['type'], axis=1, inplace=True)
        start = time.time()
        if 'xgb' in model_csv_name:
            res = model.predict(xgb.DMatrix(data))
        else:
            res = model.predict(data)
        end = time.time()
        during = end-start
        res = [round(x) for x in res]
        # 计算f1值
        score = _f1_score(res, label)
        print('F1值：%s，速率：%s 条/s，%s ms/条' % (score, round(samples / during, 5), round(during * 1000 / samples, 5)))
        return 0


def performance(min_features=False):
    rounds = 100
    if min_features:
        test_data = pd.read_csv('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/csv_keras_new/sutra')
        test_data.drop(c_20000, axis=1, inplace=True)
    else:
        test_data = pd.read_csv('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/csv_keras_new/sutra')
    total_sample = test_data.shape[0]*rounds
    model = xgb.Booster(model_file='mul_xgb_less_feature.model')
    # model = lgb.Booster(model_file='mul_lgb_test1.model')
    start_time = time.time()
    print('Started')
    for i in range(rounds):
        model.predict(xgb.DMatrix(test_data))
        # model.predict(test_data)
    total_time = time.time() - start_time
    print('速率：%s 条/s，%s ms/条' % (round(total_sample/ total_time, 5), round(total_time * 1000 / total_sample, 5)))


def main():
    start = time.time()
    print('XGB 训练开始...')
    # 多样本的家族
    base = ['suppobox', 'corebot', 'emotet', 'banjori', 'tinynuke', 'chinad', 'padcrypt', 'torpig', 'pandabanker',
            'sisron', 'bamital', 'simda', 'ramdo', 'gameover', 'dyre']
    chosed = ['feodo', 'symmi', 'volatile', 'shifu', 'bebloh', 'oderoor', 'pykspa', 'hesperbot', 'matsnu', 'fobber',
              'pushdo', 'dnsbenchmark', 'mirai', 'madmax', 'bigviktor', 'proslikefan', 'ekforward', 'ranbyus_v1',
              'murofetweekly', 'rovnix', 'sphinx', 'beebone', 'modpack', 'bedep', 'tofsee', 'szribi', 'sutra',
              'vawtrak', 'gspy', 'murofet', 'xxhex', 'omexo', 'tsifiri', 'tinba', 'gozi', 'cryptolocker']
    # X_t, X_v, y_t, y_v = load_zrz_data(clf='mul')
    # c = base + chosed
    # family_id = [family_label[x] for x in c]
    # X_t, X_v, y_t, y_v = load_new_data(clf='mul', cols=family_id)
    X_t, X_v, y_t, y_v = load_zrz_data(clf='mul', min_features=True, int_type=True)  # , min_features=True, int_type=True

    name = 'mul_xgb_tree_test'

    # 对比基分类器对预测速度的影响
    # for i in [50, 100, 150, 200]:
    #     mul_clf = xgb_mul_model(X_t, X_v, y_t, y_v, c='train', n_tree=i)
    #     mul_clf.save_model('%s.model' % name)
    #     end = time.time()
    #     dur = end-start
    #     h, m, s = int(dur//3600), int((dur % 3600)//60), int(dur % 60)
    #     print('训练用时：%s小时%s分%s秒' % (h, m, s))
    #     re_df = verification(clf='mul', zrz=True, model_csv_name=name)  # , min_features=True, int_type=True
    #     print('预测结果')
    #     print(re_df)
    #     re_df.to_csv('/home/lxf/data/DGA/training_results/xgb/%s.csv' % name+'_'+str(i), index=False)

    mul_clf = xgb_mul_model(X_t, X_v, y_t, y_v, c='train', n_tree=150)
    mul_clf.save_model('%s.model' % name)
    end = time.time()
    dur = end-start
    h, m, s = int(dur//3600), int((dur % 3600)//60), int(dur % 60)
    print('训练用时：%s小时%s分%s秒' % (h, m, s))
    re_df = verification(clf='mul', zrz=True, model_csv_name=name, min_features=True, int_type=True)  # , min_features=True, int_type=True
    print('预测结果')
    print(re_df)
    re_df.to_csv('/home/lxf/data/DGA/training_results/xgb/%s.csv' % name, index=False)


def lgb_mul_getbest():
    # lgb_test_1为78分类中准确率大于0。9的家族27个，效果很好,第二个list为后加的家族
    # murofet家族的准确率一直在0.9左右
    # 效果不好的   murofetweekly, 误报个数：98250, 正确率：0.8394
    #            enviserv,      误报个数：179603, 正确率：0.8626
    #            bebloh,        误报个数：26260, 正确率：0.7925
    #            shifu,         误报个数：516, 正确率：0.798
    #            qakbot,        误报个数：1027228, 正确率：0.676
    #            pykspa,        误报个数：357521, 正确率：0.7518
    #            nymaim,        误报个数：112287, 正确率：0.7498
    #            locky,         误报个数：123206, 正确率：0.701
    #            fobber,        误报个数：99773, 正确率：0.5011
    #            pykspa_v2,     误报个数：39497, 正确率：0.7067
    #            kraken,        误报个数：36111, 正确率：0.7296
    #            ramnit,        误报个数：52829, 正确率：0.6007
    #            dircrypt,      误报个数：26406, 正确率：0.5435
    #            bedep,         误报个数：8913, 正确率：0.6155
    #            ranbyus_v1,    误报个数：15777, 正确率：0.009
    #            sutra,         误报个数：4515, 正确率：0.5431
    #            unknownjs,     误报个数：7045, 正确率：0.2684
    #            vidro,         误报个数：4446, 正确率：0.0927
    #            pizd,          误报个数：1482, 正确率：0.3702

    # tsifiri受大数量样本影响严重
    # 加入pandabanker后tofsee准确率下降至0.7848
    # 加入szribi后tsifiri准确率变为0
    # 加入locky后necurs变为0.8628下降0.03
    lgb_test_1 = ['suppobox', 'shiotob', 'cryptolocker', 'symmi', 'corebot', 'emotet', 'banjori', 'dnschanger',
                  'sphinx', 'tinynuke', 'chinad', 'padcrypt', 'tofsee', 'torpig', 'vawtrak', 'sisron',
                  'virut', 'bamital', 'tinba', 'simda', 'ramdo', 'gameover', 'qadars', 'tsifiri',
                  'dyre'] + ['murofet', 'necurs', 'conficker', 'rovnix', 'gameoverp2p',
                             'pushdo', 'matsnu', 'xxhex', 'pandabanker', 'szribi', 'ranbyus', 'gozi', 'proslikefan',
                             'volatile', 'murofetweekly', 'enviserv']

    family_id = [family_label[x] for x in lgb_test_1]
    lgb_mul_train(c=family_id)
    save_df = verification('mul', zrz=False, c=lgb_test_1, model_csv_name='mul_lgb_test1')
    print('当前家族数量: %s' % len(lgb_test_1))


def xgb_mul_getbest():
    less_family = ['feodo', 'randomloader', 'volatile', 'shifu', 'oderoor', 'hesperbot', 'cryptowall', 'ekforward',
                   'ranbyus_v1', 'ccleaner', 'sphinx', 'tempedreve', 'geodo', 'beebone', 'dnsbenchmark', 'modpack',
                   'bedep', 'tinynuke', 'tofsee', 'szribi', 'vidro', 'torpig', 'sutra', 'gspy', 'pandabanker', 'sisron',
                   'xxhex', 'xshellghost', 'omexo', 'tsifiri', 'darkshell', 'mirai', 'pizd', 'madmax', 'blackhole',
                   'unknownjs', 'redyms', 'shiotob', 'bigviktor']
    low = ['randomloader', 'cryptowall', 'ccleaner', 'tempedreve', 'geodo', 'pykspa_v2', 'vidro', 'xshellghost']
    # 结果如下：
    base = ['suppobox', 'corebot', 'emotet', 'banjori', 'tinynuke', 'chinad', 'padcrypt', 'torpig', 'pandabanker',
            'sisron', 'bamital', 'simda', 'ramdo', 'gameover', 'dyre']
    chosed = ['feodo', 'symmi', 'volatile', 'shifu', 'bebloh', 'oderoor', 'pykspa', 'hesperbot', 'matsnu', 'fobber',
              'pushdo', 'dnsbenchmark', 'mirai', 'madmax', 'bigviktor', 'proslikefan', 'ekforward', 'ranbyus_v1',
              'murofetweekly', 'rovnix', 'sphinx', 'beebone', 'modpack', 'bedep', 'tofsee', 'szribi', 'sutra',
              'vawtrak', 'gspy', 'murofet', 'xxhex', 'omexo', 'tsifiri', 'tinba', 'gozi', 'cryptolocker']  # 新增没有影响的且效果大于0.8的家族
    add = []  # 依次添加的家族名，是所有名字中除去base的
    low_acc_family = dict()
    for i in lines_count.keys():
        if i not in base+chosed+low:
            add.append(i)
    # 依次添加一个家族
    for i in tqdm(add):
        use = base + [i] + chosed
        family_id = [family_label[x] for x in use]
        xgb_mul_train(c=family_id)
        save_df = verification('mul', zrz=False, c=use, model_csv_name='mul_xgb_test1')
        if save_df[save_df['family'] == i]['hit_ratio'].values[0] < 0.8:  # 如果新加入的家族的结果小于0.8, 继续下一个家族
            low_acc_family[i] = save_df[save_df['family'] == i]['hit_ratio'].values[0]
        else:
            flag1 = False
            for family in save_df['family'].values.tolist():
                flag = False
                if save_df[save_df['family'] == family]['hit_ratio'].values[0] < 0.8:
                    if family not in less_family:
                        flag = True
                if flag:
                    flag1 = True  # 新增家族影响其他家族效果
                    break
            if flag1:
                low_acc_family[i] = save_df[save_df['family'] == i]['hit_ratio'].values[0]
            else:
                save_df.to_csv('/home/lxf/data/DGA/training_results/mul_xgb_test1_res.csv', index=False)
                chosed.append(i)
        # elif sum(1 if x < 0.8 else 0 for x in save_df['hit_ratio'].tolist()) == 0:  # 如果新加入的家族对其他家族影响不大
        #     save_df.to_csv('/home/lxf/data/DGA/training_results/mul_xgb_test1_res.csv', index=False)
        #     chosed.append(i)
        # else:
        #     continue
    print('用到的家族：')
    print(base)
    print(chosed)
    print('效果不好的家族准确率：')
    print(low_acc_family)
    print('家族数量', len(base+chosed))


def similarity_compare():
    from utils.tools import eda
    mul_test = pd.read_csv('/data1/lxf/DGA/data/mul_78_10w.csv')
    cols = ['domain_len', '_contains_digits', '_subdomain_lengths_mean',
       '_n_grams0', '_n_grams1', '_n_grams4', '_hex_part_ratio',
       '_alphabet_size', '_shannon_entropy', '_consecutive_consonant_ratio',
       'domain_seq35', 'domain_seq36', 'domain_seq38', 'domain_seq39',
       'domain_seq40', 'domain_seq41', 'domain_seq42', 'domain_seq43',
       'domain_seq46', 'domain_seq47', 'domain_seq48', 'domain_seq49',
       'domain_seq50', 'domain_seq51', 'domain_seq52', 'domain_seq53',
       'domain_seq54', 'domain_seq55', 'domain_seq56', 'domain_seq57',
       'domain_seq58', 'domain_seq59', 'domain_seq60', 'domain_seq61',
       'domain_seq62', 'domain_seq63', 'domain_seq64', 'domain_seq65',
       'domain_seq66', 'domain_seq67', 'domain_seq68', 'domain_seq69',
       'domain_seq70', 'domain_seq71', 'domain_seq72', 'domain_seq73',
       'domain_seq74', 'domain_seq75']
    catagory = mul_test['family'].drop_duplicates().tolist()  # 类别数
    res_dict = dict()
    for i in catagory:
        tmp = mul_test[mul_test['family'] == i]
        family_name = [k for k, v in family_label.items() if v == i][0]  # family_id转换为家族名称
        res_dict[family_name] = []
        for j in cols:
            res_dict[family_name].append(tmp[j].median())
    res_df = pd.DataFrame(res_dict)
    # eda(res_df, '/home/lxf/data/DGA/training_results/', 'similarity_mean_min')
    res_df.to_csv('/home/lxf/data/DGA/training_results/similarity_median.csv', index=False)


def similarity_heatmap():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv('/home/lxf/data/DGA/training_results/similarity_median.csv')
    drop_col = ['feodo', 'randomloader', 'volatile', 'hesperbot', 'cryptowall', 'ccleaner', 'tempedreve', 'geodo', 'beebone', 'dnsbenchmark', 'modpack', 'tofsee', 'gspy', 'xshellghost', 'omexo', 'tsifiri', 'darkshell', 'mirai', 'madmax', 'blackhole', 'redyms', 'bigviktor']
    df.drop(drop_col, axis=1, inplace=True)  # 去掉样本数小于1000的家族
    corrdf = df.corr()
    corrdf.to_csv('/home/lxf/data/DGA/training_results/corr_df.csv', index=False)

    res = dict()
    for i in corrdf.columns:
        res[i] = corrdf[i].median()
    res_ = sorted(res.items(), key=lambda item: item[1], reverse=True)
    print(res_)
    print(res_[50])
    sort_name = [x[0] for x in res_]
    print(len(sort_name))
    print(sort_name[:50])
    # sns.heatmap(corrdf)
    # plt.show()


def get_txt():
    path = '/data1/lxf/DGA/'
    file_name = os.listdir(path)
    # ***多分类***
    # 在各家族抽取一些域名测试用
    # with open('/data1/lxf/DGA/data/mul_test_raw.txt', 'w') as a:
    #     for name in tqdm(file_name):
    #         with open(path+name, 'r') as r:
    #             for i in range(100):
    #                 a.write(r.readline().split(',')[0] + '\n')
    # print('多分类测试数据写入完成')

    # 二分类测试数据抽取
    # 10w全黑dga样本
    bin_data = pd.read_csv('/data1/lxf/DGA/data/all_bin_data.csv')
    print(bin_data.head())


def sometest():
    # 随便测点啥
    # 8GPU  753s, 1GPU 1820s, cpu
    import matplotlib.pyplot as plt
    model1 = lgb.Booster(model_file='bin_lgb_1434trees.model')
    model2 = xgb.Booster(model_file='bin_xgb_500trees.model')
    a = xgb.plot_tree(model2)
    plt.show()


if __name__ == '__main__':
    # xgb_mul_gridsearch()
    # lgb_mul_gridsearch()
    # xgb_mul_train()
    # lgb_mul_train(c=family_id)
    # verification('mul', zrz=False, c=col)
    # performance(min_features=True)
    new2vectors()
    # xgb_mul_gpu_train()


    # 黑样本482793




    # 利率5.4 组合贷月还5615，利息121.99w
    # 商贷利率5.4 月供7580， 利息137.9w
    # 商贷利率基准， 月供6961， 利息115.6w