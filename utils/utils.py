from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 数据集划分
def train_data_split(df, label_name='label', id_name='id', _size=0.3):
    if label_name not in df.columns or id_name not in df.columns:
        print('[ERROR] The name of either ID column or LABEL column is wrong')
        return None
    train_label = df[label_name]
    train_data = df.drop([id_name, label_name], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=_size, random_state=2020)

    return x_train, x_test, y_train, y_test


# 根据字典数据画柱状图，多用于特征重要程度图
def draw_from_dict(dicdata, RANGE=None, axis=0):
    if RANGE is None:
        RANGE = len(dicdata)
    # dicdata：字典的数据。
    # RANGE：截取显示的字典的长度。
    # axis=0，代表条状图的柱子是竖直向上的。axis=1，代表柱子是横向的。考虑到文字是从左到右的，让柱子横向排列更容易观察坐标轴。
    by_value = sorted(dicdata.items(), key=lambda item: item[1], reverse=False)
    x = []
    y = []
    for d in by_value:
        x.append(d[0])
        y.append(d[1])
    if axis == 0:
        plt.bar(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    elif axis == 1:
        plt.barh(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    else:
        return "Axis got wrong value!"