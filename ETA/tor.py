from ETA.tor_feature_extraction import TorFeatureExtraction
import json
import pandas as pd


def load_data():
    res = []
    with open('/home/lxf/data/tor/meta/tor_data.txt', 'r') as r:
        for line in r:
            res.append(json.loads(line.strip()))
    return res


def get_feature():
    features = []
    tor_feature_extraction = TorFeatureExtraction()
    data = load_data()
    for info in data:
        features.append(tor_feature_extraction.get_feature(info))
    column = [str(i) for i in range(len(features[0]))]
    features = pd.DataFrame(data=features, columns=column)
    features.fillna(0, inplace=True)
    return features


if __name__ == '__main__':
    res = get_feature()
    print(len(res))
