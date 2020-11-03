import yaml


def load_params(model, clf):
    # 加载配置文件
    with open('mul_params.yaml', 'r') as r:
        params = yaml.load(r, Loader=yaml.FullLoader)
    return params[clf][model]


if __name__ == '__main__':
    res = load_params('lightgbm', 'bin')
    print(res)
