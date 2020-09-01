import xgboost as xgb


def xgb_model(X_t, X_v, y_t, y_v, test):
    """
    :param X_t: 训练集
    :param X_v: 验证集
    :param y_t: 训练集标签
    :param y_v: 验证集标签
    :param test: 测试集数据
    :return: 返回值为对测试集预测结果
    """

    xgb_val = xgb.DMatrix(X_v, label=y_v)
    xgb_train = xgb.DMatrix(X_t, label=y_t)
    xgb_test = xgb.DMatrix(test)

    # 基本参数
    """
    reference:https://xgboost.readthedocs.io/en/latest/parameter.html
    ===基本参数(General Parameters)===
    <booster>: 默认gbtree, ['gbtree', 'gbliner', 'dart']
               gbtree: 树学习器
               dart: 带有dropout的树学习器
               gbliner: 线性学习器。一般不选用线性学习器作为booster的基学习器,因为多个线性模型的组合stacking还是一个线性模型。
    
    <verbosity>: 默认为1。打印日志的详细程度; 0: silent, 1: warning, 2: info, 3:debug
    
    <validate_parameters>: 默认为False。试验阶段参数。设为True模型则会验证参数。
    
    <nthread>: 线程数，默认为最大。
    
    <disable_default_eval_metric>: 默认为False, 是否禁用默认metric。(Flag to disable default metric. Set to 1 or true to disable.)
    
    *<num_pbuffer>: 预测缓冲区大小，xgb自动设置，无需手动设置。
    
    *<num_feature>: 特征维度大小，xgb自动设置，无需手动设置。
    
    """

    # 树学习器参数
    """    
    ===树学习器参数(Parameters for Tree Booster)===
    <eta>: 学习率, 默认为0.3
    
    <gamma>: 公式中gamma值，是叶子结点T的正则化参数，为了使T尽可能小，gamma值越大算法越保守。一般0.1、0.2这样子。
    
    <max_depth>: 构建树的深度，越大越容易过拟合，默认值为6
    
    <min_child_weight>: 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的0-1分类而言，假设 h 在 0.01 附近，
                        min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。这个参数非常影响结果，控制叶子节点中二阶
                        导的和的最小值，该参数值越小，越容易过拟合。数值越大算法越保守。
                        
    <max_delta_step>: Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is 
                      no constraint. If it is set to a positive value, it can help making the update step more 
                      conservative. Usually this parameter is not needed, but it might help in logistic regression when 
                      class is extremely imbalanced. Set it to value of 1-10 might help control the update.
                      
    <subsample>: 随机采样训练样本。[0,1]
    
    <sampling_method>: 默认uniform。[uniform, gradient_based]
                       uniform: 每个训练样本被选择的概率相等，通常subsample设置为大于0.5的值效果更好
                       gradient_based: 每个训练样本被选择概率与正则化的梯度的绝对值成正比。subsample可设置低至0.1并且不损失精度。
                       ***注意***: 仅在tree_method设置为gpu_hist时支持此采样方式。
                       
    <colsample_by*>: 对列的随机采样。 [colsample_bytree, colsample_bylevel, colsample_bynode]
                     - colsample_bytree: is the subsample ratio of columns when constructing each tree. Subsampling occurs
                     once for every tree constructed.
                     - colsample_bylevel:  is the subsample ratio of columns for each level. Subsampling occurs once for 
                     every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for
                     the current tree.
                     - colsample_bynode: is the subsample ratio of columns for each node (split). Subsampling occurs once 
                     every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the 
                     current level.
                     
    <lambda>: L2正则，数值越大越不容易过拟合。默认值为1
    
    <alpha>: L1正则，数值越大越不容易过拟合。默认值为0
    
    <tree_method>: 树构建算法。
    
    <sketch_eps>: 仅在tree_method=approx时使用
    
    <scale_pos_weight>: 用来处理正负样本不均衡的问题, 通常取：sum(negative cases) / sum(positive cases)
    
    <max_leaves>: Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.
    <updater>:  [default= grow_colmaker,prune]
                A comma separated string defining the sequence of tree updaters to run, providing a modular way to 
                construct and to modify the trees. This is an advanced parameter that is usually set automatically, 
                depending on some other parameters. However, it could be also set explicitly by a user. 
                
    <refresh_leaf>: This is a parameter of the refresh updater. When this flag is 1, tree leafs as well as tree nodes’ 
                    stats are updated. When it is 0, only node stats are updated.
                    
    <process_type>: A type of boosting process to run.
                    [default, update]
                    
    <grow_policy>: Controls a way new nodes are added to the tree.Currently supported only if tree_method is set to hist.
                   [depthwise, lossguide]
                   
    <max_bin>: Only used if tree_method is set to hist.
               Maximum number of discrete bins to bucket continuous features.
               Increasing this number improves the optimality of splits at the cost of higher computation time.
               
    <predictor>: The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
                 [auto, cpu_predictor, gpu_predictor]
    
    <num_parallel_tree>: [default=1] - Number of parallel trees constructed during each iteration. This option is used 
                         to support boosted random forest.
                         
    <monotone_constraints>: Constraint of variable monotonicity. See tutorial for more information.

    <interaction_constraints>: Constraints for interaction representing permitted interactions. The constraints must be 
                               specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]], where each inner list is 
                               a group of indices of features that are allowed to interact with each other. See tutorial
                               for more information
    """

    # 训练任务参数
    """
    ===训练任务参数(Learning Task Parameters)===
    <objective>: - reg:squarederror        平方损失回归
                 - reg:squaredlogerror     对数平方损失回归
                 - reg:logistic            逻辑回归
                 - reg:pseudohubererror    Pseudo Huber损失回归
                 - reg:gamma               log-link gamma回归，输出为gamma分布的均值
                 - reg:tweedie             log-link的Tweedie回归
                 + binary:logistic         逻辑回归二分类，输出为概率值
                 + binary:logitraw         逻辑回归二分类，输出位逻辑变换之前的分值
                 + binary:hinge            合页损失二分类，输出为0或1
                 * multi:softmax           使用softmax进行多分类，需要设置num_class
                 * multi:softprob          与softmax相同，结果包含属于每个类别的每个数据的预测概率。
                 # rank:pairwise           Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
                 # rank:ndcg               Use LambdaMART to perform list-wise ranking where Normalized Discounted 
                                           Cumulative Gain (NDCG) is maximized
                 # rank:map                Use LambdaMART to perform list-wise ranking where Mean Average Precision (MAP) 
                                           is maximized
                 $ aft_loss_distribution:  Probabilty Density Function used by survival:aft objective and aft-nloglik metric.
                 $ survival:aft            Accelerated failure time model for censored survival time data
                 $ survival:cox            Cox regression for right censored survival time data (negative values are 
                                           considered right censored).
                 $ count:poisson           poisson regression for count data, output mean of poisson distribution
    
    <eval_metric>: [rmse, rmsle, mae, mphe, logloss, error, error@t, merror, mlogloss, auc, aucpr, ndcg, map, ndcg@n, 
                    map@n, poisson-nloglik, gamma-nloglik, cox-nloglik, gamma-deviance, tweedie-nloglik, aft-nloglik,
                    interval-regression-accuracy]
    """

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        #   'gamma': 0.1,
        #   'max_depth': 8,
        #   'alpha': 0,
        'lambda': 10,
        'subsample': 0.7,
        #   'colsample_bytree': 0.5,
        #   'min_child_weight': 3,
        'eta': 0.01,  # 如同学习率
        #   'seed': 1000,
        'n_jobs': -1,  # cpu 线程数
        #   'missing': 1,
        #   'scale_pos_weight': (np.sum(y==0)/np.sum(y==1))
    }

    plst = list(params.items())
    num_rounds = 5000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    # 训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=200)
    res = model.predict(xgb_test)
    return res