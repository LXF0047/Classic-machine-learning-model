from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score


def _f1_score(predict, real, avg=None):
    if avg is None:
        avg = 'binary'
    return f1_score(real, predict, average=avg)


def _confusion_metrix(real, predict):
    return confusion_matrix(real, predict)


def _precision(real, predict):
    return precision_score(real, predict)


def _recall(real, predict):
    return recall_score(real, predict)


def _accuracy(real, predict):
    return accuracy_score(real, predict)
