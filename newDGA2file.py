#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os


def load_new_data(path):
    '''
    :param path: 从360下载的新文件
    :return: {'家族名': [uri1, uri2, uri3, ...]}
    '''
    res = dict()
    with open(path, 'r') as r:
        for i in r:
            if i.strip().startswith('#') or i.strip() == '':
                continue
            else:
                family_name = i.split('\t')[0]
                uri = i.split('\t')[1]
                if family_name not in res:
                    res[family_name] = []
                    res[family_name].append(uri)
                else:
                    res[family_name].append(uri)
    print('===新文件中家族及样本数量===')
    for family in res:
        print(family, ':', len(res[family]))
    return res, path.split('/')[-1]


def add2currentfile(new_dict, filename, log_path, path):
    '''
    :param new_dict: load_new_data()返回值
    :param filename: 日志文件名
    :param log_path: 日志文件存放位置
    :param path: dga样本文件位置
    :return:
    '''
    families = os.listdir(path)
    # 去除白样本和文件夹
    families.remove('new_family_files')
    families.remove('logs')
    with open(log_path + filename + '.log', 'w') as w:  # 日志文件存放位置
        for name in new_dict:
            if name in families:
                # 现有家族新增
                tmp = dict()  # 存现有家族的domain
                new_count = 0  # 统计新增数量
                with open(path + name, 'r') as r:
                    for i in r:
                        if i.strip() not in tmp:
                            tmp[i.strip()] = 0
                with open(path + name, 'a') as a:
                    for new_uri in new_dict[name]:
                        if new_uri not in tmp:
                            a.write(new_uri + '\n')
                            new_count += 1
                w.write('[---原有家族---]%s家族新增%s条' % (name, new_count))
                print('[---原有家族---]%s家族新增%s条' % (name, new_count))
            else:
                # 新家族写入
                with open(path + name, 'w') as w:
                    for domain in new_dict[name]:
                        w.write(domain + '\n')
                w.write('[+++新增家族+++]%s家族新增%s条' % (name, len(new_dict[name])))
                print('[+++新增家族+++]%s家族新增%s条' % (name, len(new_dict[name])))


def check_domain2():
    # 给邹哥检查txt文件里是否有dga域名
    families = os.listdir('/home/lxf/geye/data/DGA/')
    families.remove('legit_all')
    families.remove('new_family_files')
    dga_dict = dict()
    for name in families:
        with open('/home/lxf/geye/data/DGA/%s' % name, 'r') as r:
          for line in r:
              if line.strip() not in dga_dict:
                  dga_dict[line.strip()] = name
    with open('/home/lxf/geye/data/DGA/new_family_files/check_dga.txt', 'r') as r:
        with open('/home/lxf/geye/data/DGA/new_family_files/logs/dga_domains2.txt', 'w') as w:
            for i in r:
                if i.strip() in dga_dict:
                    w.write(dga_dict[i.strip()] + ':' + i)


def main():
    # 360网址：https://data.netlab.360.com/dga/
    # 下载链接：https://data.netlab.360.com/feeds/dga/dga.txt
    path = '/home/lxf/geye/data/DGA/new_family_files/new_data/360new20201017.txt'  # 从360下载的文件位置
    log_path = '/home/lxf/geye/data/DGA/new_family_files/logs/'  # 日志文件存放位置
    dga_path = '/home/lxf/geye/data/DGA/'
    res, name = load_new_data(path)
    print('=' * 100)
    add2currentfile(res, name, log_path, dga_path)


if __name__ == '__main__':
    main()