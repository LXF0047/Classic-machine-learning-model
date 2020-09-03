from sklearn.model_selection import train_test_split


def train_data_split(df, label_name='label', id_name='id', _size=0.3):
    if label_name not in df.columns or id_name not in df.columns:
        print('[ERROR] The name of either ID column or LABEL column is wrong')
        return None
    train_label = df[label_name]
    train_data = df.drop([id_name, label_name], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=_size, random_state=2020)

    return x_train, x_test, y_train, y_test