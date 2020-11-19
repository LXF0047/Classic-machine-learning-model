import pandas as pd
import numpy as np
from sklearn.svm import SVC
import re
from utils.tools import eda
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score


def base_info(t='train'):
    cat = ['oplocdistrict', 'industryphy', 'industryco', 'dom', 'enttype', 'enttypeitem', 'oploc', 'state',
           'orgid', 'jobid', 'adbusign', 'townsign', 'regtype']  #
    if t == 'train':
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/base_info.csv')
    else:
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/test/base_info.csv')
    # 100%缺失列：['ptbusscope', 'midpreindcode']
    data.drop(['ptbusscope', 'midpreindcode'], axis=1, inplace=True)
    # 筛选经营范围为投资的
    unknown = ['opscope']
    # tz_col = []
    # rgx = re.compile('创业投资|保障性住房|对非上市企业|对未上市企业')
    # for index, item in data.iterrows():
    #     if re.search(rgx, item['opscope']):
    #         tz_col.append(1)
    #     else:
    #         tz_col.append(0)
    # data['keyword_in_opscope'] = tz_col
    data.drop(['opscope'], axis=1, inplace=True)

    # 将详细地址变为粗略地址, 53类
    data['dom'] = data['dom'].map(lambda x: x[:10])
    # 经营场所
    data['oploc'] = data['oploc'].map(lambda x: x[:10])
    # 将起始时间变为经营时长, 删除原有opfrom和opto列
    data['opfrom'] = pd.to_datetime(data['opfrom']).dt.year.astype('int')
    data['opyears'] = 2020 - data['opfrom']
    # 20201116新增将结束时间变为还有多久结束  线下0.8155909265783258 线上0.821
    # data['opto'].fillna('1990/1/1', inplace=True)
    # data['opto'] = pd.to_datetime(data['opto']).dt.year.astype('int')
    # data['left_years'] = data['opto'] - 2020
    data.drop(['opfrom', 'opto'], axis=1, inplace=True)
    # 经营人数填充缺失值0
    data['empnum'].fillna(0, inplace=True)
    # 删除缺失值多的列
    # compform    9784   值只有1,2，和空值 效果下降
    # parnum    13195
    # exenum    14132
    # opform    9964   下降
    # enttypeminu    10085
    # protype    14845
    # reccap    11883    实缴资本
    # forreccap    14826
    # forregcap    14808
    # congro    14809

    # 对缺失的尝试
    # 20201115新增 venind提升明显0.82988042000 三折交叉验证f1值0.8026493854479493
    data['venind'].fillna('0', inplace=True)
    data['venind'] = data['venind'].astype('str')
    cat = cat + ['venind']

    # 废弃20201116新增 compform 线下0.8086850098390341 线上0.81620950307 下降了
    # data['compform'].fillna('0', inplace=True)
    # data['compform'] = data['compform'].astype('str')
    # cat = cat + ['compform']

    # 废弃20201117新增实缴资本类别交了1没交0缺失-1， 线下0.8173819324710285  线上0.82085174357
    # data['reccap'].fillna('na', inplace=True)
    # data['reccap'] = data['reccap'].astype('str')
    # data['reccap_cat'] = data['reccap'].apply(lambda x: 'neg' if x == '0.0' else ('na' if x == 'na' else 'pos'))
    # cat = cat + ['reccap_cat']

    # 废弃20201117新增 opform处理  线下0.8090691301480373 线上0.81361617319
    # data['opform'].fillna('na', inplace=True)
    # cat = cat + ['opform']

    # 类别标签转为int,效果骤减
    # for item in cat:
    #     data[item] = LabelEncoder().fit_transform(data[item])

    # 删除模型重要程度为0的特征
    # data.drop(['oplocdistrict', 'industryco', 'orgid', 'jobid', 'adbusign', 'oploc'], axis=1, inplace=True)

    # 删除缺失和效果不好的基本特征
    data.drop(['parnum', 'exenum', 'enttypeminu', 'protype', 'reccap', 'opform',
               'forreccap', 'forregcap', 'congro', 'compform'], axis=1, inplace=True)

    return data, cat


def annual_report_info(t='train'):
    # 3336家公司有企业年报信息
    # cat = ['emp_sign']
    cat = []

    if t == 'train':
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/annual_report_info.csv')
    else:
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/test/annual_report_info.csv')

    # 企业年报年度个数
    year_count = data.groupby(['id'])['ANCHEYEAR'].count().reset_index(name='year_count')

    # 20201117新增
    # EMPNUMSIGN是否公示不同年份取均值，缺失值用0补充，合并到base后缺失值用-1补充  预测出974个 线下0.8064449778114495  线上0.81863103426
    # data['EMPNUMSIGN'].fillna(0, inplace=True)
    # tmp1 = data.groupby(['id'])['EMPNUMSIGN'].mean().reset_index(name='emp_sign')

    # 公示状态PUBSTATE  目前是按照均值做的，可以拆成不同列来表示状态  预测出945个  线下0.8092665836765024  线上0.81924896087
    # data['PUBSTATE'].fillna(0, inplace=True)
    # tmp2 = data.groupby(['id'])['PUBSTATE'].mean().reset_index(name='pub_state')

    # 是否对外投资FORINVESTSIGN  目前是按照均值做的，可以拆成不同列来表示状态  预测出933个 线下0.8054878669608566  线上0.82022339366
    # data['FORINVESTSIGN'].fillna(0, inplace=True)
    # tmp3 = data.groupby(['id'])['FORINVESTSIGN'].mean().reset_index(name='for_invest')
    # res = tmp3[['id', 'for_invest']]


    return year_count, cat


def change_info(t='train'):
    # 8726家公司有变更信息
    if t == 'train':
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/change_info.csv')
    else:
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/test/change_info.csv')
    # 变更次数
    change_times = data.groupby(['id'])['bgxmdm'].count().reset_index(name='change_times')
    return change_times


def news_info():
    # 927家公司有舆情信息
    cat = []
    data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/news_info.csv')
    # data['news_mean'] = data['positive_negtive'].map(lambda x: 1 if x == '积极' else (0 if x == '中立' else -1))
    # 评价平均数
    # news_mean = data.groupby(['id'])['news_mean'].mean().reset_index(name='news_mean')

    # 消极评价出现次数
    print(data['id'].nunique())
    data['news_mean'] = data['positive_negtive'].map(lambda x: 1 if x == '消极' else 0)
    neg_count = data.groupby(['id'])['news_mean'].sum().reset_index(name='neg_count')
    print(neg_count['neg_count'])


    # return news_mean, cat


def other_info():
    # 一共包括1888家公司的信息
    data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/other_info.csv')
    print(data['id'].nunique())
    return data


def tax_info(t):
    cat = []
    if t == 'train':
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/tax_info.csv')
    else:
        data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/test/tax_info.csv')
    # 平均税额
    # tax_mean = data.groupby(['id'])['TAX_AMOUNT'].mean().reset_index(name='tax_mean')

    # TAX_ITEMS中包括罚款的次数
    # fakuan_count = {}
    # for index, line in data.iterrows():
    #     if '罚款' in line['TAX_ITEMS']:
    #         if line['id'] not in fakuan_count:
    #             fakuan_count[line['id']] = 1
    #         else:
    #             fakuan_count[line['id']] += 1
    # fakuan_d = {'id': [], 'fakuan_count': []}
    # for i in fakuan_count:
    #     fakuan_d['id'].append(i)
    #     fakuan_d['fakuan_count'].append(fakuan_count[i])
    # fakuan_df = pd.DataFrame(fakuan_d)

    return data, cat


def label():
    data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/entprise_info.csv')
    return data


def merge_df(t='train'):
    _label = label()
    # beta0.1 使用base和change两个表的信息
    base_data, base_cat = base_info(t)
    change_data = change_info(t)
    train_data = pd.merge(base_data, change_data, on='id', how='outer')
    train_data['industryco'].fillna(0, inplace=True)
    train_data['enttypeitem'].fillna(0, inplace=True)
    train_data['regcap'].fillna(0, inplace=True)
    train_data['change_times'].fillna(0, inplace=True)

    # 加入企业年报信息
    # annual_data, annual_cat = annual_report_info(t)
    # train_data = pd.merge(train_data, annual_data, on='id', how='outer')
    # train_data['emp_sign'].fillna(-1, inplace=True)
    # train_data['pub_state'].fillna(-1, inplace=True)
    # train_data['for_invest'].fillna(-1, inplace=True)

    # 加入税务信息
    # tax_data, tax_cat = tax_info(t)
    # train_data = pd.merge(train_data, tax_data, on='id', how='outer')
    # train_data['fakuan_count'].fillna(-1, inplace=True)

    cat_feature = base_cat

    for c in cat_feature:
        if isinstance(train_data[c][10], float):
            train_data[c] = train_data[c].astype('str')

    if t == 'train':
        train_ = pd.merge(train_data, _label, on='id', how='outer')
        train_.drop(['id'], axis=1, inplace=True)
        return train_, cat_feature
    elif t == 'cv':
        train_ = pd.merge(train_data, _label, on='id', how='outer')
        return train_
    else:
        train_data.drop(['id'], axis=1, inplace=True)
        return train_data


# /home/lxf/projects/auto_boosting/competitions/illegal_fund_raising/predict_res
def save_res(name, pre):
    if name is None:
        name = 'res'
    res_id = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/test/base_info.csv')['id'].tolist()
    pre = [round(x) for x in pre]
    res_dict = {'id': res_id, 'score': pre}
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv('/home/lxf/projects/auto_boosting/competitions/illegal_fund_raising/predict_res/%s.csv' % name, index=False)
    print('结果文件%s保存成功' % name)


def cb_train():
    from machine_learning.ensemble_learning.boosting import BoostingModules
    train_data, cat_feature = merge_df(t='train')
    cb_m = BoostingModules(train_data).cb_model(category_cols=cat_feature)

    # load test data
    test_data = merge_df(t='test')
    test_data.fillna(0, inplace=True)
    res = cb_m.predict(test_data)
    print('预测结果中非法集资企业个数：%s' % sum([round(x) for x in res]))
    # save result file
    # save_res('best_add_annual_forinvest', res)
    # 线下交叉验证
    # cross_validation(cb_m, cv_iter=3)


def cross_validation(model, cv_iter=5):
    train_data, cat_feature = merge_df(t='train')
    _label = train_data['label']
    train_data.drop(['label'], axis=1, inplace=True)
    score = cross_val_score(model, train_data, _label, scoring='f1', cv=cv_iter, verbose=0, n_jobs=-1)
    print('%s折交叉验证结果：%s' % (cv_iter, np.mean(score)))


def full_info_companies():
    # 只有42家公司所有信息都有, 这42家公司中23家为非法集资公司
    label_ = label()
    files = ['base_info', 'annual_report_info', 'change_info', 'news_info', 'other_info', 'tax_info']
    company_id = dict()
    for file in files:
        companies = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/%s.csv' % file)['id'].drop_duplicates().tolist()
        print('%s表中有%s个公司的数据，其中%s家为非法集资公司，占比%s' % (file, len(companies),
                                                   label_[label_['id'].isin(companies)]['label'].sum(), round(label_[label_['id'].isin(companies)]['label'].sum()*100/len(companies), 2)))
        if file not in company_id:
            company_id[file] = companies
    full_info = list(set(company_id['base_info']).intersection(company_id['annual_report_info'],
                                                               company_id['change_info'], company_id['news_info'],
                                                               company_id['other_info'], company_id['tax_info']))
    print('信息全的公司有%s家，其中%s家为非法集资公司。' % (len(full_info), label_[label_['id'].isin(full_info)]['label'].sum()))


def get_black():
    base_ = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/base_info.csv')
    annual_ = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/annual_report_info.csv')
    news_ = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/news_info.csv')
    change_ = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/change_info.csv')
    other_ = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/other_info.csv')
    tax_ = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/tax_info.csv')

    label_ = label()
    neg_label = label_[label_['label'] == 1]['id'].tolist()

    d = {'neg_base_info': base_, 'neg_annual_report_info': annual_, 'neg_news_info': news_, 'neg_change_info': change_,
         'neg_other_info': other_, 'neg_tax_info': tax_}

    for i in d:
        neg_tmp = d[i][d[i]['id'].isin(neg_label)]
        neg_tmp.to_csv('/home/lxf/projects/competion/illegal_fund/data/train/%s.csv' % i, index=False)


if __name__ == '__main__':
    cb_train()
    # full_info_companies()
    # rf_train()
    # news_info()
    # annual_report_info(t='test')
    # tax_info(t='train')