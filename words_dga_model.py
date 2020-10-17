from __future__ import division
import pandas as pd
import wordninja
import xgboost as xgb
import re
import wordsegment as ws
# from pybloomfilter import BloomFilter
import lightgbm as lgb
from utils.utils import train_data_split
from utils.score import _f1_score
import enchant
from tqdm import tqdm


ws.load()


def load_black_data():
    file_path = '/home/lxf/data/DGA/word_dga/feature/'
    file_names = ['gozi', 'suppobox', 'gozi']
    data = pd.read_csv(file_path + file_names[0])
    for file in file_names[1:]:
        # print(file)
        tmp = pd.read_csv(file_path + file)
        pd.concat([data, tmp])
    data.drop(['family'], axis=1, inplace=True)
    print('DGA黑样本%s条读取完成' % data.shape[0])
    return data


def load_white_data(n=1667269):
    data = pd.read_csv('/home/lxf/data/DGA/word_dga/feature/legit_words', nrows=n)
    data.drop(['family'], axis=1, inplace=True)
    print('DGA白样本%s条读取完成' % data.shape[0])
    return data


def pinyin_tokens(text):
    pinyin_re = re.compile("[^aoeiuv]?h?[iuv]?(ai|ei|ao|ou|er|ang?|eng?|ong|a|o|e|i|u|ng|n)?")
    tokens = []
    sub_len = len(text)
    while sub_len > 0:
        match = pinyin_re.search(text)
        tokens.append(text[match.start():match.end()])
        text = text[match.end():]
        sub_len -= match.end() - match.start()
    return tokens


def meaningprob_dga(text, to=None, length=None):
    enpinyin_bf = BloomFilter.open("/home/soft/resource/aimatrix/dga/model/dga_model_en_pinyin.bf")
    _text_length = len(text)
    if _text_length <= 0:
        return 0.0

    if isinstance(text, (list, tuple, set)):
        words = text
        if length is None:
            _text_length = sum(map(len, words))
        else:
            _text_length = length
    elif to == "py":  # pinyin participle
        words = pinyin_tokens(text)
    elif to == "en":  # english participle
        words = ws.segment(text)  # Attention: f.s -> fs,  x.jr->xjr
    else:
        raise Exception("Not valid parameters.")

    if len(words) <= 0:
        return 0.0

    m_count, m_len = 0, 0
    # print text, words
    for word in words:
        if len(word) > 1 and (word in enpinyin_bf or word[1:] in enpinyin_bf or word[2:] in enpinyin_bf):
            # print "===",word
            m_count += 1
            m_len += len(word)

    if _text_length <= 0:
        _text_length = 1
    prob1 = (m_count / len(words) + m_len / _text_length) / 2
    prob2 = m_len / _text_length
    prob = max(prob1, prob2)
    # print("%s participle result: %s, meaningprob: %f" % (to, " /".join(words), prob))
    return prob


def re_split(sentence):
    # 两种方式递归分词
    if isinstance(sentence, str):
        tmp1 = wordninja.split(sentence)
        return re_split(tmp1)
    else:
        tmp2 = [ws.segment(x) for x in sentence]
        tmp3 = [j for i in tmp2 for j in i]
        if sorted(sentence) == sorted(tmp3):
            return tmp3
        else:
            return re_split(tmp3)


def xgb_model(X_t, X_v, y_t, y_v):
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
    num_rounds = 1500
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_rounds, watchlist,
                      early_stopping_rounds=50)
    return model


def lgb_model(X_t, X_v, y_t, y_v):
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
    }
    lgb_train = lgb.Dataset(X_t, y_t)
    lgb_eval = lgb.Dataset(X_v, y_v, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=1500, valid_sets=lgb_eval, early_stopping_rounds=50)
    return gbm


def train(model_name, sample_percent):
    '''
    :param model_name: 保存的模型文件名
    :param sample_percent: 采样时白样本是黑样本的几倍，最大7倍
    :return:
    '''
    # black 235785
    # white 1667269
    # 加载训练集
    print('保存模型名称: %s, 黑白样本采样比例(黑 : 白) : 1:%s' % (model_name, sample_percent))
    black = load_black_data()
    black['label'] = 1
    white = load_white_data(n=sample_percent*black.shape[0])
    white['label'] = 0
    print('[训练集] 黑样本数量：%s，白样本数量：%s' % (black.shape[0], white.shape[0]))
    train_data = pd.concat([black, white])
    X_t, X_v, y_t, y_v = train_data_split(train_data)
    # 训练
    xgb_m = xgb_model(X_t, X_v, y_t, y_v)
    lgb_m = lgb_model(X_t, X_v, y_t, y_v)
    # 保存模型
    xgb_m.save_model('/home/lxf/data/DGA/word_dga/modules/xgb_%s.model' % model_name)
    lgb_m.save_model('/home/lxf/data/DGA/word_dga/modules/lgb_%s.model' % model_name)


def verification(model_name):
    # 验证模型效果
    xgb_m = xgb.Booster(model_file='/home/lxf/data/DGA/word_dga/modules/xgb_%s.model' % model_name)
    lgb_m = lgb.Booster(model_file='/home/lxf/data/DGA/word_dga/modules/lgb_%s.model' % model_name)
    black_ = load_black_data()
    # 预测所有黑样本
    xgb_b_r = [round(x) for x in xgb_m.predict(xgb.DMatrix(black_))]
    lgb_b_r = [round(x) for x in lgb_m.predict(black_)]
    xgb_black = (sum(xgb_b_r) / len(xgb_b_r))
    lgb_black = (sum(lgb_b_r) / len(lgb_b_r))
    # 预测所有白样本
    white_ = load_white_data(n=1667268)
    xgb_w_r = [round(x) for x in xgb_m.predict(xgb.DMatrix(white_))]
    lgb_w_r = [round(x) for x in lgb_m.predict(white_)]
    xgb_white_ = (1 - sum(xgb_w_r) / len(xgb_w_r))
    lgb_white_ = (1 - sum(lgb_w_r) / len(lgb_w_r))

    print('[黑样本准确率] XGB: %s, LGB: %s' % (xgb_black, lgb_black))
    print('[白样本准确率] XGB: %s, LGB: %s' % (xgb_white_, lgb_white_))


def main():
    model_name = ''
    sample_percent = 1
    train(model_name, sample_percent)
    verification(model_name)


if __name__ == '__main__':
    # train()
    load_black_data()
    load_white_data()
    # matsnu 0.024512701880649444
    # suppobox 0.37164895045857477
    # gozi 0.77

    # 原来RF模型对白样本的误报0.133590            0.04478
    # xgb白样本误报0.000670                     0.0007
    # lgb白样本误报0.013980                     0.00644

    # 3000lgb白样本误报0.011320          0.01473
    # 3000xgb白样本误报0.000670          0.00073


    # 3000trees的准确率
    # [黑样本准确率]
    # XGB: 0.9680707428568928, LGB: 0.9663562177102764
    # [白样本准确率]
    # XGB: 0.9944855895992726, LGB: 0.9945821547585632

    # 1000trees准确率
    # [黑样本准确率]
    # XGB: 0.9358297941480456, LGB: 0.9387109333681418
    # [白样本准确率]
    # XGB: 0.9897293056665155, LGB: 0.9902061336269874

