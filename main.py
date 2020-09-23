# 用于其他在10服务器上的验证
import os
from tqdm import tqdm
import Levenshtein


def test():
    path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/families/'
    file_names = os.listdir(path)

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
    hard2detect_path = '/data1/lxf/DGA/hard2detect.txt'
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

    res2 = pd.read_csv('/home/lxf/data/DGA/training_results/mul_xgb_78_res.csv')
    print(res2[res2['hit_ratio'] > 0.9]['family'].tolist())


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
    res = [x if x <200000 else 200000 for x in lines_count.values()]
    print(sum(res)/2440000)



if __name__ == '__main__':
    res_compare()

