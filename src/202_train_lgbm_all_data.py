
import gc
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import time
import warnings

from contextlib import contextmanager
from datetime import datetime, timedelta
from glob import glob
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold, GroupKFold
from tqdm import tqdm

from utils import line_notify, to_json, rmse, save2pkl, submit
from utils import FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2, CAT_COLS
from utils import CustomTimeSeriesSplitter

#==============================================================================
# Train LightGBM with Simple Hold Out
#==============================================================================

warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Display/plot feature importance
def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # for checking all importance
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# Train LightGBM
def train_lightgbm(train_df,test_df,debug=False):
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # set data structure
    lgb_train = lgb.Dataset(train_df[feats],
                            label=train_df['demand'],
                            free_raw_data=False)

    # params optimized by optuna
    params ={
            'device' : 'gpu',
            'gpu_use_dp':True,
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'poisson',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 5,
            'max_leaves':int(.7*5** 2),
            'colsample_bytree': 1.0,
            'subsample': 0.9,
            'reg_lambda': 1,
            'reg_alpha': 0,
            'min_child_weight': 1,
            'verbose': -1,
            'seed':326,
            'bagging_seed':326,
            'drop_seed':326
            }

    # train model
    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train],
                    verbose_eval=100,
                    num_boost_round=500,
                    )

    # save model
    reg.save_model('../output/lgbm_all_data.txt')

    # save predictions
    oof_preds += reg.predict(train_df[feats], num_iteration=reg.best_iteration)
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration)

    # save feature importances
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
    fold_importance_df["fold"] = 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    del reg
    gc.collect()

    # save out of fold prediction
    train_df.loc[:,'demand'] = oof_preds
    train_df[['id','d','demand']].to_csv(oof_file_name, index=False)

    # display importances
    display_importances(feature_importance_df,
                        '../imp/lgbm_importances_holdout.png',
                        '../imp/feature_importance_lgbm_holdout.csv')

    # Full RMSE score and LINE Notify
    full_rmse = rmse(train_df['demand'], oof_preds)
    line_notify('Full RMSE score %.6f' % full_rmse)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

def main(debug=False):
    with timer("Load Datasets"):
        # load feathers
        files = sorted(glob('../feats/*.feather'))
        df = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)], axis=1)

        # use selected features
        df = df[configs['features']]

        # drop old data
        df = df[df['date']>'2014-04-25']

        # split train & test
        #=======================================================================
        # 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
        # 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
        # 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)
        #=======================================================================

        train_df = df[df['date']<'2016-04-25']
        test_df = df[df['date']>='2016-04-25']

        del df
        gc.collect()

    with timer("Run LightGBM with kfold"):
        train_lightgbm(train_df, test_df, debug=debug)

if __name__ == "__main__":
    oof_file_name = "../output/oof_lgbm_holdout.csv"
    configs = json.load(open('../configs/202_lgbm_all_data.json'))
    with timer("Full model run"):
        main(debug=False)