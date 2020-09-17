import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def _xgboost(params):
    return xgb.XGBClassifier(params)