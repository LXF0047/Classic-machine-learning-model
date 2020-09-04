from sklearn.ensemble import RandomForestClassifier
from utils.utils import train_data_split
from utils.score import _f1_score


def rf_model(train):
    # 参数
    '''
     n_estimators=100,            树的个数
     criterion="gini",            分裂方式  [entropy, gini]
     max_depth=None,              树最大深度
     min_samples_split=2,         节点分裂所需的最小样本数
     min_samples_leaf=1,          叶节点所需的最小样本数
     min_weight_fraction_leaf=0., The minimum weighted fraction of the sum total of weights (of all
                                  the input samples) required to be at a leaf node. Samples have
                                  equal weight when sample_weight is not provided.
     max_features="auto",         寻找最佳分裂时需要的样本数量
     max_leaf_nodes=None,
     min_impurity_decrease=0.,
     min_impurity_split=None,
     bootstrap=True,
     oob_score=False,
     n_jobs=None,
     random_state=None,
     verbose=0,
     warm_start=False,
     class_weight=None,
     ccp_alpha=0.0,
     max_samples=None
    '''

    rfc = RandomForestClassifier(random_state=2020, verbose=1)
    x_t, x_v, y_t, y_v = train_data_split(train)
    rfc.fit(x_t, y_t)
    pre = rfc.predict(x_v)
    print('Random Forest Model F1 Score: %s' % _f1_score(pre, y_v))
    print(rfc.feature_importances_)
    return rfc
