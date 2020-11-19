import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})
# myfont = FontProperties(fname=r'/Users/Library/Fonts/SourceHanSansSC-Normal.otf')
# sns.set(font=myfont.get_family())
# sns.set_style("whitegrid", {"font.sans-serif": ['Source Han Sans CN']})


def load_data(file='base_info'):
    label_df = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/entprise_info.csv')
    data = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/%s.csv' % file)
    train_id = label_df['id'].tolist()
    # 训练集为1，测试集为0
    data['train_test'] = data['id'].apply(lambda x: 'train' if x in train_id else 'test')
    return data


def null_count(data, col):
    '''
    :param data: 数据集
    :param col: 需要统计的列
    :return: 缺失值个数
    '''
    tmp = data.isnull().sum(axis=0).reset_index(name='null_count')
    null_amount = tmp[tmp['index'] == col]['null_count'].values[0]
    print('%s列缺失值个数：%s，占比%s%%' % (col, null_amount, round(100*(null_amount/data[col].shape[0]), 2)))
    return null_amount


def plot_discrete(data, _y, _hue):
    '''
    :param data:输入的数据集
    :param _y: y轴上显示的类别
    :param _hue: 根据哪一列分类
    :return: None
    '''
    # 离散型画柱状图
    plt.figure(figsize=(15, 10))
    sns.countplot(y=_y, hue=_hue, data=data, palette="Greens_d")
    plt.title(_y)
    plt.show()


def plot_continuous(data, col):
    sns.set_style('whitegrid', {'font.sans-serif': ['simhei', 'Arial']})
    print(data['train_test'].value_counts())
    g = sns.kdeplot(data[col][(data['train_test'] == 'train')], color="Red", shade=True)  # 训练集
    g = sns.kdeplot(data[col][(data["train_test"] == 'test')], color="Green", shade=True)  # 测试集

    # g = sns.kdeplot(trains[column], ax=g, color="Green", shade=True)# trains中所有
    # g = sns.kdeplot(tests[column], ax =g, color="Blue", shade= True)# test中所有

    g.set_xlabel(col)
    g.set_ylabel("Frequency")
    g = g.legend(["train_data", "test_data"])

    plt.show()

def base_discrete_analysis(data):
    # oplocdistrict
    # plot_discrete(data, 'oplocdistrict', 'train_test')

    # industryphy
    # plot_discrete(data, 'industryphy', 'train_test')

    # industryco 将细分类别变为大类取除以1000的余数，缺失值用10000补充
    # data['industryco'].fillna(10000, inplace=True)
    # data['industryco'] = data['industryco'].apply(lambda x: x//1000)
    # plot_discrete(data, 'industryco', 'train_test')

    # dom  取前10位
    # data['dom'] = data['dom'].apply(lambda x: x[:10])
    # plot_discrete(data, 'dom', 'train_test')

    # enttype
    # plot_discrete(data, 'enttype', 'train_test')

    # 企业类型小类 enttypeitem 缺失8214 取除以1000的余数，缺失值用10000补充
    # data['enttypeitem'].fillna(10000, inplace=True)
    # data['enttypeitem'] = data['enttypeitem'].apply(lambda x: x//1000)
    # plot_discrete(data, 'enttypeitem', 'train_test')

    # state公示状态 无缺失
    # plot_discrete(data, 'state', 'train_test')

    # orgid 机构标识 无缺失
    # plot_discrete(data, 'orgid', 'train_test')

    # jobid 职位标识 无缺失 类似id的数值类型
    # data['jobid'] = data['jobid'].astype('str')
    # data['jobid'] = data['jobid'].apply(lambda x: x[:10])
    # plot_discrete(data, 'jobid', 'train_test')

    # adbusign 是否广告经营 无缺失 0-1二类
    # plot_discrete(data, 'adbusign', 'train_test')

    # townsign 是否城镇 无缺失 0-1二类
    # plot_discrete(data, 'townsign', 'train_test')

    # regtype 主题登记类型 1-3-4类别特征
    # plot_discrete(data, 'regtype', 'train_test')

    # empnum 从业人数 数值类型特征  缺失值个数：5250，占比21% 存在极大值 可分箱
    # data['empnum'].fillna(0, inplace=True)
    # plot_continuous(data, 'empnum')

    # compform 经营方式 1-2类别特征
    # data['compform'].fillna(0, inplace=True)
    # plot_discrete(data, 'compform', 'train_test')

    # parnum 合伙人数 缺失值个数：22526，占比91%
    # data['parnum'].fillna(0, inplace=True)
    # plot_continuous(data, 'parnum')

    # exenum 执行人数 失值个数：23487，占比94.4581%  有异常值100000
    # data['exenum'].fillna(0, inplace=True)
    # data['exenum'] = data['exenum'].apply(lambda x: 0 if x == 100000.0 else x)
    # plot_continuous(data, 'exenum')

    # opform 类别特征 缺失值个数：15865，占比63.8%
    # data['opform'].fillna('unknown', inplace=True)
    # plot_discrete(data, 'opform', 'train_test')

    # ptbusscope 兼营范围 训练集全部缺失

    # venind 类别特征 缺失：16428 66.07%  这个效果贼好提升0.009
    null_count(data, 'venind')
    data['venind'].fillna(0, inplace=True)
    plot_discrete(data, 'venind', 'train_test')

    # enttypeminu 企业类型细类
    null_count(data, 'enttypeminu')
    data['enttypeminu'].fillna(, inplace=True)
    plot_discrete(data, 'venind', 'train_test')

def some_test():
    train = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/train/base_info.csv')
    test = pd.read_csv('/home/lxf/projects/competion/illegal_fund/data/test/base_info.csv')
    # null_count(train, 'exenum')
    # null_count(test, 'exenum')
    test['exenum'] = test['exenum'].apply(lambda x: 0 if x == 100000.0 else x)
    print(test['exenum'].describe())


if __name__ == '__main__':
    # 数据为训练加测试的整体
    data = load_data()
    base_discrete_analysis(data)
    # some_test()