from __future__ import division
import pandas as pd
import wordninja
import xgboost as xgb
import re
import wordsegment as ws
from pybloomfilter import BloomFilter
import enchant
from tqdm import tqdm


ws.load()


def load_black_data():
    file_path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/families/'
    file_names = ['gozi']  # , 'suppobox', 'gozi'
    res = []
    for name in file_names:
        with open(file_path + name, 'r') as r:
            for line in r:
                res.append(line.split('.')[0])
    return res


def load_handel_white_data():
    # path_1 = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv'  # 训练白样本
    # path_2 = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/top1w.txt'  # top1w白名单
    path_3 = '/home/lxf/data/DGA/words_white_dga.txt'
    with open(path_3, 'r') as r:
        white = r.readlines()
    return white


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



if __name__ == '__main__':
    # data = load_black_data()
    # d = enchant.Dict("en_US")
    # not_word_count = 0
    # for i in tqdm(data):
    #     # if not ws.segment(i) == wordninja.split(i):
    #     #     print(i, '>>>ws: ', ws.segment(i), '>>>nj: ', wordninja.split(i))
    #     split = re_split(i)
    #     res = [1 if not d.check(x) else 0 for x in split]
    #     if sum(res) != 0:
    #         # 存在无意义的词
    #         not_word_count += 1
    # print(not_word_count/len(data))

    load_handel_white_data()

    # matsnu 0.024512701880649444
    # suppobox 0.37164895045857477
    # gozi 0.77


