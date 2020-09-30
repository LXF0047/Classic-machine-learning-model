import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pandas as pd

d = pd.read_csv('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/new_feature/legit.csv').head(50)
print(d.astype('int'))
