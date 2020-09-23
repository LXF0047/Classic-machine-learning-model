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
                    'suppobox': 78}
# 邹哥采样的多分类数据样本家族对应关系
zrz_family_label = {'virut': 0, 'gameover': 1, 'murofet': 2, 'dnschanger': 3, 'dyre': 4, 'ranbyus': 5,
                    'murofetweekly': 6, 'banjori': 7, 'rovnix': 8, 'gameoverp2p': 9, 'chinad': 10, 'emotet': 11,
                    'vawtrak': 12, 'symmi': 13, 'corebot': 14, 'gozi': 15, 'qadars': 16, 'tinba': 17, 'pushdo': 18,
                    'padcrypt': 19, 'pykspa_v2': 20, 'kraken': 21, 'bamital': 22, 'simda': 23, 'bebloh': 24,
                    'matsnu': 25, 'ramdo': 26, 'dircrypt': 27, 'torpig': 28, 'sphinx': 29, 'bedep': 30, 'szribi': 31,
                    'sisron': 32, 'tinynuke': 33, 'sutra': 34, 'pandabanker': 35, 'vidro': 36, 'xxhex': 37, 'shifu': 38,
                    'pizd': 39, 'ekforward': 40, 'oderoor': 41, 'volatile': 42, 'blackhole': 43, 'geodo': 44,
                    'tempedreve': 45, 'mirai': 46}


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
    :return:
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
        bin_data = pd.read_csv('/data1/lxf/DGA/data/all_bin_data.csv')
        return train_data_split(bin_data, label_name='type')
    elif clf == 'mul':
        if cols is not None:
            new_id = [x for x in range(len(cols))]
            new_dict = dict(zip(cols, new_id))
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


def load_zrz_data(clf='bin'):
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
        print('数据集读取完成')
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


def xgb_bin_model(X_t, X_v, y_t, y_v):
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
    num_rounds = 5000  # 迭代次数,cv10折交叉验证结果
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    # model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)# , feval=f1_score, maximize=True
    # return model
    cv_res = xgb.cv(plst, xgb_train, num_rounds, nfold=10, early_stopping_rounds=100,
                   maximize=True, seed=2020, verbose_eval=2)

    return cv_res.shape[0]


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
        'num_threads': 28
        # 'device': 'gpu',
        # 'gpu_platform_id': 1,
        # 'gpu_device_id': 1,
    }
    if c == 'train':
        gbm = lgb.train(lgb_params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval,
                        early_stopping_rounds=100)
        return gbm
    else:
        n_est = lgb.cv(lgb_params, lgb_train, num_boost_round=5000, nfold=5, early_stopping_rounds=200, seed=2020,
                       verbose_eval=2)
        return n_est.shape[0]


def xgb_mul_model(X_t, X_v, y_t, y_v, c='train'):
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
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': n_family,
        'eval_metric': 'mlogloss',
        'verbosity': 2,
        'tree_method': 'gpu_hist',
        'gpu_id': 1
    }
    plst = list(xgb_params.items())
    num_rounds = 5000  # 迭代次数
    if c == 'train':
        watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
        model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
        return model
    elif c == 'cv':
        cv_res = xgb.cv(plst, xgb_train, num_rounds, nfold=5, early_stopping_rounds=100,
                        maximize=True, seed=2020, verbose_eval=2)
        return cv_res.shape[0]


def xgb_mul_model_gpu(client, x, y):
    # 使用多个GPU,报错ImportError: Dask needs to be installed in order to use this module
    # 没整明白。。
    # 参考https://xgboost.readthedocs.io/en/latest/gpu/index.html
    n_family = len(set(y.tolist()))
    dtrain = DaskDMatrix(client, x, y)
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             'booster': 'gbtree',
                             'objective': 'multi:softmax',
                             'num_class': n_family,
                             'eval_metric': 'mlogloss',
                             # Golden line for GPU training
                             'tree_method': 'gpu_hist'},
                            dtrain,
                            num_boost_round=5000, evals=[(dtrain, 'train')])
    bst = output['booster']
    history = output['history']

    # you can pass output directly into `predict` too.
    prediction = xgb.dask.predict(client, bst, dtrain)
    prediction = prediction.compute()
    print('Evaluation history:', history)
    return prediction


def xgb_mul_gpu_train():
    # 多CPU并行训练，没弄明白
    X_t, X_v, y_t, y_v = load_new_data(clf='mul')
    with LocalCUDACluster(n_workers=8, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            xgb_mul_model_gpu(client, X_t, y_t)


def xgb_bin_train():
    # 使用邹哥采样的训练数据，加载带id和label的二分类数据集，将数据集分为训练用和测试用
    # train_set, test_set = load_bin_data(topn=150000)
    # test_feature, test_label = train_handel(test_set)
    # X_t, X_v, y_t, y_v = train_data_split(train_set)
    # 使用所有数据的数据集
    X_t, X_v, y_t, y_v = load_new_data(clf='bin')
    xgb_clf = xgb_bin_model(X_t, X_v, y_t, y_v)
    print(xgb)
    # xgb_res = xgb_clf.predict(xgb.DMatrix(test_feature))
    # xgb_res = [round(x) for x in xgb_res]
    # xgb_f1_score = _f1_score(xgb_res, test_label)
    # print('XGB预测结果：%s' % xgb_f1_score)


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
    n_es = lgb_mul_model(X_t, X_v, y_t, y_v)
    print(n_es)

    # param_test1 = {
    #     'max_depth': range(3, 8, 2),
    #     'min_child_weight': range(2, 6, 2)
    # }
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=n_es, max_depth=6,
    #                                                 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #                                                 objective='binary:logistic', nthread=-1, scale_pos_weight=1,
    #                                                 seed=2020),
    #                         param_grid=param_test1, scoring='roc_auc', n_jobs=-1, cv=10, verbose=2)
    # gsearch1.fit(X_t, y_t)
    # print('n_estimators = %s\nmax_depth = %s\nmin_child_weight = %s\nbest_score_ = %s' % (n_es,
    #                                                                                       gsearch1.best_params_[
    #                                                                                           'max_depth'],
    #                                                                                       gsearch1.best_params_[
    #                                                                                           'min_child_weight'],
    #                                                                                       gsearch1.best_score_))


def get_new_family():
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
    # 将新家族样本转化为向量csv文件
    path1 = '/home/lxf/data/DGA/new_family_raw/'
    path2 = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/families/'
    new1 = ['shiotob', 'bigviktor', 'enviserv']  # 新家族
    new2 = ['qakbot', 'conficker', 'necurs', 'cryptolocker', 'locky', 'suppobox']  # 没参加训练的家族
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
            res = extract_feature([x.split(',')[0] for x in r])
            np.savetxt('/home/lxf/data/DGA/new_family_raw/%s' % name, res, delimiter=',')
        tmp = pd.read_csv('/home/lxf/data/DGA/new_family_raw/%s' % name)
        tmp.columns = cols
        tmp['family'] = name
        tmp.to_csv('/home/lxf/data/DGA/new_family/%s' % name, index=False)
        print('%s保存成功，数量：%s' % (name, tmp.shape[0]))


def verification(clf, zrz=False, c=None):
    model_csv_name = 'mul_lgb_test1'
    # 邹哥模型用到的家族
    file_path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/csv_keras_new/'
    file_names = os.listdir(file_path)

    # 新加入的家族
    new_file_path = '/home/lxf/data/DGA/new_family/'  # 以前没有参加训练的样本
    new_file_names = os.listdir(new_file_path)
    # 白样本
    # legit_path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv'
    # legit_data = pd.read_csv(legit_path)
    # legit_data.drop(['type'], axis=1, inplace=True)

    # 加载训练好的模型
    if 'xgb' in model_csv_name:
        model = xgb.Booster(model_file='%s.model' % model_csv_name)  # 使用邹哥采样的数据重训练的xgb多分类模型，未进行调参
    elif 'lgb' in model_csv_name:
        model = lgb.Booster(model_file='%s.model' % model_csv_name)
    else:
        print('Wrong name')
        return 0

    # 进行预测
    if clf == 'mul':
        # pid = os.getpid()
        total_time = 0  # 预测总用时
        total_sample = 0  # 预测样本总条数

        # 写入结果csv中内容
        res_df = {'family': [], 'unhit': [], 'total': [], 'hit_ratio': []}
        print('开始预测')
        if c is not None:
            _filenames = c
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
                else:
                    continue
            else:
                if name in new_file_names:
                    tmp = pd.read_csv(new_file_path + name)
                    tmp.drop(['family'], axis=1, inplace=True)
                else:
                    tmp = pd.read_csv(file_path + name)

            pre_start = time.time()  # 开始预测时间

            if 'xgb' in model_csv_name:
                res = model.predict(xgb.DMatrix(tmp))  # XGB预测结果
            elif 'lgb' in model_csv_name:
                lgb_res = model.predict(tmp)
                res = [list(x).index(max(x)) for x in lgb_res]  # LGB预测结果
            else:
                print('Wrong File Name!')
                return 0

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

            print('家族：%s, 误报个数：%s, 正确率：%s' % (name, false_count, round(1 - false_count/tmp.shape[0], 4)))
            # print('PID: %s , Used Memory: %s MB' % (pid, psutil.Process(pid).memory_info().rss / 1024 / 1024))

            # 存入结果csv
            res_df['family'].append(name)
            res_df['unhit'].append(false_count)
            res_df['total'].append(tmp.shape[0])
            res_df['hit_ratio'].append(round(1 - false_count/tmp.shape[0], 4))

        df2save = pd.DataFrame(res_df)
        # df2save.to_csv('/home/lxf/data/DGA/training_results/%s_res.csv' % model_csv_name, index=False)
        print('速率：%s 条/s，%s ms/条' % (round(total_sample/total_time, 5), round(total_time*1000/total_sample, 5)))
        return df2save


def performance():
    rounds = 1000

    test_data = pd.read_csv('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/csv_keras_new/sutra')
    total_sample = test_data.shape[0]*rounds
    # model = xgb.Booster(model_file='mul_xgb_zrz.model')
    model = lgb.Booster(model_file='mul_lgb_33.model')
    start_time = time.time()
    print('Started')
    for i in range(rounds):
        # model.predict(xgb.DMatrix(test_data))
        model.predict(test_data)
    total_time = time.time() - start_time
    print('速率：%s 条/s，%s ms/条' % (round(total_sample/ total_time, 5), round(total_time * 1000 / total_sample, 5)))


def main():
    # xgb_bin_train()
    # xgb二分类调参, 尝试不同数据量对模型调参结果的影响
    # xgb_bin_gridsearch()
    start = time.time()
    # xgb_mul_train()
    lgb_mul_train()
    end = time.time()
    dur = end-start
    h, m, s = int(dur//3600), int((dur % 3600)//60), int(dur % 60)
    print('用时：%s小时%s分%s秒' % (h, m, s))


def check_gpu():
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))


def test1():
    # 选取部分列进行训练
    # lbg_col_43 = ['necurs', 'suppobox', 'shiotob', 'conficker', 'enviserv', 'cryptolocker', 'symmi', 'volatile', 'bebloh',
    #        'matsnu', 'corebot', 'pushdo', 'emotet', 'banjori', 'murofetweekly', 'rovnix', 'dnschanger', 'sphinx',
    #        'modpack', 'tinynuke', 'chinad', 'padcrypt', 'tofsee', 'szribi', 'torpig', 'vawtrak', 'gspy', 'pandabanker',
    #        'sisron', 'murofet', 'virut', 'xxhex', 'bamital', 'omexo', 'tsifiri', 'tinba', 'simda', 'gameoverp2p',
    #        'ramdo', 'unknownjs', 'gameover', 'qadars', 'dyre']
    # lgb_col_33 = ['necurs', 'suppobox', 'conficker', 'enviserv', 'cryptolocker', 'symmi', 'bebloh', 'matsnu', 'corebot',
    #        'pushdo', 'emotet', 'banjori', 'murofetweekly', 'rovnix', 'dnschanger', 'sphinx', 'tinynuke', 'chinad',
    #        'padcrypt', 'szribi', 'torpig', 'vawtrak', 'sisron', 'murofet', 'virut', 'bamital', 'tinba', 'simda',
    #        'gameoverp2p', 'ramdo', 'gameover', 'qadars', 'dyre']
    # xgb_test1为78分类中准确率大于0.9的家族
    # xgb_test1 = ['suppobox', 'corebot', 'emotet', 'banjori', 'tinynuke', 'chinad', 'padcrypt', 'torpig', 'pandabanker',
    #              'sisron', 'bamital', 'simda', 'ramdo', 'gameover', 'dyre']
    xgb_lgb_col_50 = ['dircrypt', 'ramnit', 'enviserv', 'cryptolocker', 'fobber', 'shiotob', 'symmi', 'bedep', 'suppobox', 'locky', 'bebloh', 'murofet', 'pizd', 'qakbot', 'tinba', 'qadars', 'vawtrak', 'dnschanger', 'kraken', 'simda', 'nymaim', 'necurs', 'chinad', 'sisron', 'torpig', 'sutra', 'ranbyus_v1', 'vidro', 'padcrypt', 'oderoor', 'corebot', 'ramdo', 'pykspa_v2', 'pykspa', 'conficker', 'pushdo', 'sphinx', 'ranbyus', 'szribi', 'banjori', 'emotet', 'gozi', 'rovnix', 'proslikefan', 'pandabanker', 'shifu', 'virut', 'unknownjs', 'ekforward', 'xxhex']
    family_id = [family_label[x] for x in xgb_lgb_col_50]  # 选取列的id号
    # xgb_mul_train(c=family_id)
    lgb_mul_train(c=family_id)
    verification('mul', zrz=False, c=xgb_lgb_col_50)


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
    # 加入pandabanker后tofsee准确率下降至0.7848
    # 加入szribi后tsifiri准确率变为0
    # 加入locky后necurs变为0.8628下降0.03
    lgb_test_1 = ['suppobox', 'shiotob', 'cryptolocker', 'symmi', 'corebot', 'emotet', 'banjori', 'dnschanger',
                  'sphinx', 'tinynuke', 'chinad', 'padcrypt', 'tofsee', 'torpig', 'vawtrak', 'sisron',
                  'virut', 'bamital', 'tsifiri', 'tinba', 'simda', 'ramdo', 'gameover', 'qadars',
                  'dyre'] + ['murofet', 'necurs', 'conficker', 'rovnix', 'gameoverp2p',
                             'pushdo', 'matsnu', 'xxhex', 'pandabanker', 'szribi', 'ranbyus', 'gozi', 'proslikefan',
                             'pykspa_v2']

    family_id = [family_label[x] for x in lgb_test_1]
    lgb_mul_train(c=family_id)
    verification('mul', zrz=False, c=lgb_test_1)
    print('当前家族数量: %s' % len(lgb_test_1))


def xgb_mul_getbest():
    base = ['suppobox', 'corebot', 'emotet', 'banjori', 'tinynuke', 'chinad', 'padcrypt', 'torpig', 'pandabanker',
            'sisron', 'bamital', 'simda', 'ramdo', 'gameover', 'dyre']
    chosed = []
    add = []
    for i in lines_count.keys():
        if i not in base:
            add.append(i)
    for i in add:
        use = base + [i] + chosed
        family_id = [family_label[x] for x in use]
        xgb_mul_train(c=family_id)
        save_df = verification('mul', zrz=False, c=use)
        try:
            pd_res = pd.read_csv('/home/lxf/data/DGA/training_results/mul_xgb_test1_res.csv')['hit_ratio']
            check = sum(1 if x < 0.8 else 0 for x in pd_res.tolist())
            if check == 0:
                save_df.to_csv('/home/lxf/data/DGA/training_results/mul_xgb_test1_res.csv', index=False)
                chosed.append(i)
            else:
                continue
        except:
            save_df.to_csv('/home/lxf/data/DGA/training_results/mul_xgb_test1_res.csv', index=False)
    print(base)
    print(chosed)








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


if __name__ == '__main__':
    # main()
    # verification('mul')
    # xgb_mul_gridsearch()
    # lgb_mul_gridsearch()
    # xgb_mul_train()
    # lgb_mul_train(c=family_id)
    # verification('mul', zrz=False, c=col)
    # performance()
    # similarity_heatmap()
    xgb_mul_getbest()
    # lgb_mul_getbest()
    # xgb_bin_train()