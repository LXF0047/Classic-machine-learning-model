import numpy as np


def f1_score(preds, dtrain):
    # xgboost自定义F1值metric
    label = dtrain.get_label()
    preds = 1.0/(1.0+np.exp(-preds))
    pred = [int(i >= 0.5) for i in preds]
    tp = sum([int(i == 1 and j == 1) for i, j in zip(pred, label)])
    precision = float(tp)/sum(pred)
    recall = float(tp)/sum(label)
    return 'f1-score', 2 * (precision*recall/(precision+recall))
