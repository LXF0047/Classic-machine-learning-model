# 用于其他在10服务器上的验证
import os
from tqdm import tqdm
from utils.utils import isEnglish
import Levenshtein


def test():
    path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/families/'
    # file_names = os.listdir(path)
    file_names = ['matsnu', 'suppobox', 'gozi']

    top1w = dict()

    with open('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/top1w.txt', 'r', encoding='latin1') as r:
        for i in r:
            top1w[i.split(',')[0].split('.')[0]] = 0
    print('读取top1w完成')

    white_d = dict()
    with open('/home/lxf/projects/test/white_dga.txt', 'r') as r:
        for i in r:
            white_d[i.strip()] = 0

    print('读取白名单完成')
    hard2detect_path = '/data1/lxf/DGA/hard2detect_words.txt'
    for file in file_names:
        with open(path + file, 'r') as r:
            for i in tqdm(r):
                if 8 < len(i.split(',')[0].split('.')[0]) < 20:
                    if i.split(',')[0] not in white_d:
                        for seem in top1w.keys():
                            flag = False
                            if Levenshtein.distance(i.split(',')[0].split('.')[0], seem) <= 3:
                                print(i.strip(), '  >>> ', seem)
                                flag = True
                                with open(hard2detect_path, 'a') as a:
                                    a.write(i)
                            if flag:
                                break
            print('文件:%s处理完成' % file)


def test3():
    # 找容易误报的白样本
    long = []
    short = []
    top1w = dict()
    with open('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/top1w.txt', 'r', encoding='latin1') as r:
        for i in r:
            top1w[i.split(',')[0]] = 0
    print('读取top1w完成')
    for line in top1w.keys():
        if len(line.split('.')[0]) > 15 and isEnglish(line.split('.')[0]):
            long.append(line)
        elif len(line.split('.')[0]) <= 15 and isEnglish(line.split('.')[0]):
            short.append(line)
        else:
            continue

    with open('/data1/lxf/DGA/white_long.txt', 'w') as long_file:
        for i in long:
            long_file.write(i + '\n')

    with open('/data1/lxf/DGA/white_short.txt', 'w') as short_file:
        for i in short:
            short_file.write(i + '\n')


# 生成dga对抗样本用
def new_dga_sample(company, hard=True):
    # 生成dga对抗数据用
    # 基于新选取的黑白域名
    import random
    path = '/data1/lxf/DGA/'
    black1 = []  # 其他类型dga  都用  2949
    black2 = []  # 单词类型dga  都用  40257
    black3 = []  # 容易检出的dga  差多少到5w用多少
    white1 = []  # 长的白域名  16876
    white2 = []  # 短的白域名
    domain_list = dict()
    with open('/home/lxf/data/DGA/word_dga/raw/pykspa3000', 'r') as hard1:  # path + 'hard2detect.txt'
        for i in hard1:
            black1.append(i.split(',')[0] + '\n')
    with open(path + 'hard2detect_words.txt', 'r') as hard2:
        for i in hard2:
            black2.append(i.split(',')[0] + '\n')
    with open('/home/lxf/data/DGA/word_dga/raw/conficker6000', 'r') as easy:  # path + 'easy2detect.txt'
        for i in easy:
            black3.append(i.strip() + '\n')
    with open(path + 'white_long.txt', 'r') as long:
        for i in long:
            white1.append(i)
    with open(path + 'white_short.txt', 'r') as short:
        for i in short:
            white2.append(i)
    with open('/home/lxf/data/intel/domain.list', 'r') as intel:
        for line in intel:
            domain_list[line.split(',')[0]] = 0

    if hard:
        _black = []
        # for i in range(37051):
        #     _black.append(random.choice(black2))
        # for i in range(6000):
        #     _black.append(random.choice(black3))
        _white = []
        for i in range(33124):
            w_ = random.choice(white2)
            if w_ not in domain_list:
                _white.append(w_)
            else:
                continue
        # res = black1 + _black + white1 + _white
        black_res = black1 + black3 + black2  # 2949+6000+40257=49206
        white_res = white1 + _white
        print('白样本去重数量： %s， 黑样本去重数量：%s' % (len(set(white_res)), len(set(black_res))))
        print('总样本数量： %s' % len(black_res + white_res))
        with open(path + company + '_hard_black_new.txt', 'w') as w:
            for i in black_res:
                w.write(i)
        with open(path + company + '_hard_white_new.txt', 'w') as w:
            for i in white_res:
                w.write(i)
    else:
        _black = []
        for i in range(1000):
            _black.append(random.choice(black1))
            _black.append(random.choice(black2))
        for i in range(48000):
            _black.append(random.choice(black3))
        _white = []
        for i in range(50000):
            _white.append(random.choice(white2))
        print(len(_black + _white))
        with open(path + company + '_easy_black.txt', 'w') as w:
            for i in _black:
                w.write(i)
        with open(path + company + '_easy_white.txt', 'w') as w:
            for i in _white:
                w.write(i)
    print('保存完成')


def handel_white_dga():
    # 处理白名单
    tmp = []
    with open('/home/lxf/projects/test/whitedga.txt', 'r') as r:
        for i in r:
            tmp.append(i.split('\t')[1])
    with open('/home/lxf/projects/test/white_dga.txt', 'a') as a:
        for i in tmp:
            a.write(i+'\n')


def res_compare():
    import pandas as pd
    from bin_retrain import zrz_family_label
    res = pd.read_csv('/home/lxf/data/DGA/training_results/mul_lgb_78_res.csv')
    # names = [x for x in zrz_family_label.keys()]
    # print(res[res['family'].isin(names)][['family', 'hit_ratio']])
    low_score = ['murofetweekly', 'enviserv', 'tofsee', 'omexo', 'bebloh']
    lgb_test_1 = ['suppobox', 'shiotob', 'cryptolocker', 'symmi', 'corebot', 'emotet', 'banjori', 'dnschanger',
                  'sphinx', 'tinynuke', 'chinad', 'padcrypt', 'tofsee', 'torpig', 'vawtrak', 'sisron',
                  'virut', 'bamital', 'tsifiri', 'tinba', 'simda', 'ramdo', 'gameover', 'qadars',
                  'dyre'] + ['murofet', 'necurs', 'conficker', 'rovnix', 'gameoverp2p',
                             'pushdo', 'matsnu', 'xxhex']
    drop_col = low_score + lgb_test_1
    a = res[~(res['family'].isin(drop_col))]
    # print(res[res['family'].isin(a.tolist())]['family', 'total', 'hit'])
    # print(a.tolist())
    # print(len(a))
    print(a.sort_values(by='total'))


def test2():
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
                   'simda': 132233, 'gameoverp2p': 418000, 'ramnit': 132319, 'pizd': 2353, 'madmax': 181,
                   'ramdo': 104000,
                   'dircrypt': 57845, 'blackhole': 732, 'kraken': 133533, 'nymaim': 448743, 'gozi': 235786,
                   'ranbyus': 785648, 'unknownjs': 9630, 'redyms': 34, 'gameover': 5000000, 'qadars': 222088,
                   'dyre': 1381889,
                   'shiotob': 8003, 'bigviktor': 999, 'enviserv': 1306928, 'qakbot': 3170167, 'conficker': 1789506,
                   'necurs': 5992235, 'cryptolocker': 1786999, 'locky': 412003, 'suppobox': 130294}
    # res = [x if x <200000 else 200000 for x in lines_count.values()]
    # print(sum(res)/2440000)
    a = []
    for i in lines_count:
        if lines_count[i] > 10000:
            a.append(i)
    print(a)
    print(len(a))


def importance_plot():
    from utils.utils import draw_from_dict
    a = {'domain_len': 6279, 'domain_seq73': 10022, 'domain_seq72': 9223, '_consecutive_consonant_ratio': 5691, 'domain_seq67': 94589, 'domain_seq70': 96710, '_shannon_entropy': 42463, '_alphabet_size': 7127, 'domain_seq48': 4176, 'domain_seq47': 2787, 'domain_seq46': 2080, 'domain_seq49': 4367, '_contains_digits': 3249, 'domain_seq65': 86783, 'domain_seq51': 5879, 'domain_seq59': 29944, 'domain_seq74': 8458, '_n_grams0': 43467, 'domain_seq53': 6966, 'domain_seq50': 5133, 'domain_seq58': 27426, 'domain_seq60': 44062, 'domain_seq61': 51278, 'domain_seq41': 164, 'domain_seq75': 8398, '_subdomain_lengths_mean': 13695, 'domain_seq71': 90791, 'domain_seq55': 11007, 'domain_seq42': 307, 'domain_seq57': 25398, 'domain_seq69': 97455, 'domain_seq56': 18680, 'domain_seq38': 105, 'domain_seq40': 98, 'domain_seq62': 63069, 'domain_seq63': 70534, 'domain_seq64': 83036, 'domain_seq54': 8257, 'domain_seq43': 454, 'domain_seq66': 95478, 'domain_seq68': 95939, 'domain_seq52': 6775, '_n_grams1': 4684, '_n_grams4': 3050, '_hex_part_ratio': 391, 'domain_seq35': 64, 'domain_seq39': 115, 'domain_seq36': 100}
    by_value = sorted(a.items(), key=lambda item: item[1], reverse=False)
    low = []
    for i in a:
        if a[i] < 1000:
            low.append(i)
    # print(by_value)
    rank_list = [x[0] for x in by_value]
    print(rank_list)
    print(len(a.keys()), len(low))
    # draw_from_dict(a)


def check_intel_domain():
    # 检查情报中是否有白样本
    intel = dict()
    with open('/home/lxf/data/intel/domain.list', 'r') as r:
        for line in r:
            intel[line.split(',')[0]] = 0
    with open('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/top1w.txt', 'r', encoding='latin1') as r:
        for i in r:
            if i.split(',')[0] in intel:
                print(i.split(',')[0])


if __name__ == '__main__':
    # res_compare()
    # test2()
    # importance_plot()
    new_dga_sample('B', hard=True)
    # check_intel_domain()