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


if __name__ == '__main__':
    test()

