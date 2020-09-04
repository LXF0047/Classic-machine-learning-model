from sklearn.metrics import f1_score, confusion_matrix


def _f1_score(predict, real, avg=None):
    if avg is None:
        avg = 'binary'
    return f1_score(real, predict, average=avg)


def _confusion_metrix(real, predict):
    return confusion_matrix(real, predict)