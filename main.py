# 用于其他在10服务器上的验证
import os
from tqdm import tqdm
import Levenshtein

def test():
    words_dga = ['matsnu', 'suppobox', 'gozi']
    path = '/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/families/'
    file_names = os.listdir(path)

    others = [x for x in file_names if x not in words_dga]

    tmp = []  # 存三个家族和长度在范围内的
    res = []  # 存tmp在白名单过滤后的
    print('读取非拼音DGA域名')
    for file in tqdm(others):
        with open(path + file, 'r') as r:
            for i in r:
                if 8 < len(i.split(',')[0].split('.')[0]) < 20:
                    tmp.append(i.split(',')[0])
                    # print(i.split(',')[0])
    print('读取DGA域名')
    for file in words_dga:
        with open(path + file, 'r') as r:
            for i in r:
                tmp.append(i.strip())

    # 读白名单
    print('读取白名单')
    white_d = dict()
    with open('/home/lxf/projects/test/white_dga.txt', 'r') as r:
        for i in r:
            white_d[i.strip()] = 0

    for i in tmp:
        if i not in white_d:
            res.append(i)

    # 读top1m
    top1w = []
    final = []
    print('读取top10000')
    with open('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/top1w.txt', 'r', encoding='latin1') as r:
        for i in r:
            try:
                top1w.append(i.split(',')[0].split('.')[0])
            except UnicodeDecodeError as e:
                print(e)
    print('开始对比，并记录五万条')
    with open('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/high_imitation.txt', 'w') as w:
        flag = 0
        for i in tqdm(res):
            for j in top1w:
                if Levenshtein.distance(i.split('.')[0], j) <= 3:
                    print(i.split('.')[0], ' >>> ', j)
                    flag += 1
                    w.write(i)
                    break
            if flag > 50000:
                break
        print('高仿域名写入完成')



    # print(len(tmp), len(res))


def handel_white_dga():
    tmp = []
    with open('/home/lxf/projects/test/whitedga.txt', 'r') as r:
        for i in r:
            tmp.append(i.split('\t')[1])
    with open('/home/lxf/projects/test/white_dga.txt', 'a') as a:
        for i in tmp:
            a.write(i+'\n')



if __name__ == '__main__':
    test()
