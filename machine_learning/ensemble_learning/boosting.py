import xgboost as xgb
import lightgbm as lgb
from ngboost.ngboost import NGBoost
from catboost import CatBoostClassifier
from utils.utils import train_data_split
from ngboost import NGBClassifier
from ngboost.distns import k_categorical, Bernoulli


class BoostingModules(object):
    def __init__(self, train_data):
        self.rounds = 2000
        self.early_stop = 10
        self.X_t, self.X_v, self.y_t, self.y_v = train_data_split(train_data)
        self.modelname = None

    def xgb_model(self, params):
        xgb_val = xgb.DMatrix(self.X_v, label=self.y_v)
        xgb_train = xgb.DMatrix(self.X_t, label=self.y_t)

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

        plst = list(params.items())
        num_rounds = self.rounds  # 迭代次数
        watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

        model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=self.early_stop)
        # print(model.get_fscore())

        if self.modelname is not None:
            model.save_model(self.modelname + '_xgb.model')
        return model

    def lgb_model(self, params):
        lgb_train = lgb.Dataset(self.X_t, self.y_t)
        lgb_eval = lgb.Dataset(self.X_v, self.y_v, reference=lgb_train)

        gbm = lgb.train(params, lgb_train, num_boost_round=self.rounds, valid_sets=lgb_eval,
                        early_stopping_rounds=self.early_stop)
        # print(gbm.feature_importance())
        if self.modelname is not None:
            gbm.save_model(self.modelname + '_lgb.model')
        return gbm

    def cb_model(self, category_cols=None):
        if category_cols is None:
            category_cols = []
        category_id = []  #
        for index, value in enumerate(self.X_t.columns):
            if value in category_cols:
                category_id.append(index)
        model = CatBoostClassifier(iterations=self.rounds, learning_rate=0.1, cat_features=category_id, loss_function='Logloss',
                                   logging_level='Verbose', eval_metric='AUC')
        model.fit(self.X_t, self.y_t, eval_set=(self.X_v, self.y_v), early_stopping_rounds=self.early_stop)
        # res = model.predict_proba(self.test)[:, 1]
        importance = model.get_feature_importance(prettified=True)  # 显示特征重要程度
        print(importance)
        if self.modelname is not None:
            model.save_model(self.modelname + '_cb.model')
        return model

    def ng_model(self):
        ngb_cat = NGBClassifier(Dist=k_categorical(2), verbose=True)
        ng_clf = ngb_cat.fit(self.X_t, self.y_t)
        print(ng_clf.feature_importances_)
        return ng_clf

    # xgb多gpu训练准备并行训练数据
    def load_higgs_for_dask(self, client, X_t, X_v, y_t, y_v):
        from xgboost.dask import DaskDMatrix
        '''
        :param client: gpu设备
        :param X_t: 训练集
        :param X_v: 验证集
        :param y_t: 训练集标签
        :param y_v: 验证集标签
        :return: dask.datafram格式的数据
        '''
        import dask.dataframe as dd
        # 1. Create a Dask Dataframe from Pandas Dataframe.
        ddf_higgs_train = dd.from_pandas(X_t, npartitions=8)
        ddf_higgs_test = dd.from_pandas(X_v, npartitions=8)
        ddf_y_train = dd.from_pandas(y_t, npartitions=8)
        ddf_y_test = dd.from_pandas(y_v, npartitions=8)
        # 2. Create Dask DMatrix Object using dask dataframes
        ddtrain = DaskDMatrix(client, ddf_higgs_train, ddf_y_train)
        ddtest = DaskDMatrix(client, ddf_higgs_test, ddf_y_test)

        return ddtrain, ddtest

    def xgb_mul_gpu_train(self):
        import time
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        from utils.utils import second2hms
        '''
        :param X_t: train
        :param X_v: test
        :param y_t: train label
        :param y_v: test label
        :return: fitted model
        '''
        # https://xgboost.readthedocs.io/en/latest/gpu/index.html  xgb官方文档
        # https://examples.dask.org/machine-learning/xgboost.html  dask官方文档
        # https://towardsdatascience.com/lightning-fast-xgboost-on-multiple-gpus-32710815c7c3  案例
        # https://gist.github.com/MLWhiz/44d39ab3a01fe4e57c974133276705f9  数据集并行处理方式
        # pip install fsspec>=0.3.3

        n_family = len(set(self.y_t.tolist()))

        with LocalCUDACluster(n_workers=8) as cluster:
            with Client(cluster) as client:
                print('数据集并行化处理')
                ddtrain, ddtest = self.load_higgs_for_dask(client, self.X_t, self.X_v, self.y_t, self.y_v)
                param = {'objective': 'binary:logistic',
                         'eta': 0.1,
                         'eval_metric': 'auc',
                         'verbosity': 2,
                         'tree_method': 'gpu_hist',
                         }
                # 'nthread': -1
                print("多GPU训练开始 ...")
                tmp = time.time()
                output = xgb.dask.train(client, param, ddtrain, num_boost_round=10000, evals=[(ddtest, 'test')])
                multigpu_time = time.time() - tmp
                print('训练完成')
                bst = output['booster']
                multigpu_res = output['history']
                h, m, s = second2hms(multigpu_time)
                print("Multi GPU Training Time: %s h %s m %s s" % (h, m, s))
        return bst
