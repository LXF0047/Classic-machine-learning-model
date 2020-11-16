import pandas as pd
from sklearn.svm import SVC
import re
from utils.tools import eda
from sklearn.preprocessing import LabelEncoder


def base_info(t='train'):
    cat = ['oplocdistrict', 'industryphy', 'industryco', 'dom', 'enttype', 'enttypeitem', 'oploc', 'state',
           'orgid', 'jobid', 'adbusign', 'townsign', 'regtype', 'venind'
           ]  #
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
    print(data['oploc'].nunique())
    # 将起始时间变为经营时长, 删除原有opfrom和opto列
    data['opfrom'] = pd.to_datetime(data['opfrom']).dt.year.astype('int')
    data['opyears'] = 2020 - data['opfrom']
    data.drop(['opfrom', 'opto'], axis=1, inplace=True)
    # 经营人数填充缺失值0
    data['empnum'].fillna(0, inplace=True)
    # 删除缺失值多的列
    # compform    9784
    # parnum    13195
    # exenum    14132
    # opform    9964  # 略微下降
    # enttypeminu    10085
    # protype    14845
    # reccap    11883
    # forreccap    14826
    # forregcap    14808
    # congro    14809

    # 对缺失的尝试
    # venind提升明显0.8298
    data['venind'].fillna('0', inplace=True)
    data['venind'] = data['venind'].astype('str')
    cat = cat + ['venind']

    # 类别标签转为int
    # for item in cat:
    #     data[item] = LabelEncoder().fit_transform(data[item])

    data.drop(['compform', 'parnum', 'opform', 'exenum', 'enttypeminu', 'protype', 'reccap',
               'forreccap', 'forregcap', 'congro'], axis=1, inplace=True)

    return data, cat


def base_opscope(df):
    import re
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    words = []
    for i in df:
        tmp1 = i.replace('（依法须经批准的项目，经相关部门批准后方可开展经营活动）', '')
        result_list = re.split(pattern, tmp1)
        print(result_list)


def annual_report_info():
    # 3336家公司有企业年报信息
    cat = []
    data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/annual_report_info.csv')
    # 企业年报年度个数
    year_count = data.groupby(['id'])['ANCHEYEAR'].count().reset_index(name='year_count')
    print(data['id'].nunique())
    return data


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


def tax_info():
    cat = []
    data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/tax_info.csv')
    # 平均税额
    tax_mean = data.groupby(['id'])['TAX_AMOUNT'].mean().reset_index(name='tax_mean')

    return tax_mean, cat


def label():
    data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/entprise_info.csv')
    return data


def merge_df(t='train'):
    _label = label()
    # beta0.1 使用base和change两个表的信息
    base_data, cat_feature = base_info(t)
    change_data = change_info(t)
    train_data = pd.merge(base_data, change_data, on='id', how='outer')
    train_data['industryco'].fillna(0, inplace=True)
    train_data['enttypeitem'].fillna(0, inplace=True)
    train_data['regcap'].fillna(0, inplace=True)
    train_data['change_times'].fillna(0, inplace=True)
    for c in cat_feature:
        if isinstance(train_data[c][10], float):
            train_data[c] = train_data[c].astype('int')

    if t == 'train':
        train_ = pd.merge(train_data, _label, on='id', how='outer')
        train_.drop(['id'], axis=1, inplace=True)
        return train_, cat_feature
    else:
        train_data.drop(['id'], axis=1, inplace=True)
        return train_data


# /home/lxf/projects/auto_boosting/competitions/illegal_fund_raising/predict_res
def save_res(name, pre):
    if name is None:
        name = 'res'
    res_id = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/test/base_info.csv')['id'].tolist()

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
    print(sum([round(x) for x in res]))

    # save result file
    save_res('best_add_venind_3', res)


def rf_train():
    from machine_learning.ensemble_learning.random_forest import rf_model
    train_data, cat_feature = merge_df(t='train')
    rf_m = rf_model(train_data)
    # load test data
    test_data = merge_df(t='test')
    test_data.fillna(0, inplace=True)
    res = rf_m.predict(test_data)
    print(sum([round(x) for x in res]))

    # save result file
    save_res('best_add_venind_rf', res)


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