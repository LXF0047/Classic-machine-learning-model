import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

xgb.Booster.load_model('bin_xgb.model')
