
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from utils import line_notify, to_json, rmse, save2pkl, submit
from utils import NUM_FOLDS, FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2
from utils import CustomTimeSeriesSplitter

#==============================================================================
# Train LightGBM
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

# LightGBM GBDT with Group KFold
def kfold_lightgbm(train_df, test_df, num_folds, debug=False):
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))

    # Cross validation
    folds = TimeSeriesSplit(n_splits=num_folds)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['demand'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['demand'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params ={
                'device' : 'gpu',
                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'max_depth': 5,
                'colsample_bytree': 1.0,
                'subsample': 0.9,
                'reg_lambda': 1,
                'reg_alpha': 0,
                'min_child_weight': 1,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        # train model
        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        reg.save_model('../output/lgbm_'+str(n_fold)+'.txt')

        # save predictions
        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        # save feature importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSE score and LINE Notify
    full_rmse = rmse(train_df['demand'], oof_preds)
    line_notify('Full RMSE score %.6f' % full_rmse)

    # display importances
    display_importances(feature_importance_df,
                        '../imp/lgbm_importances.png',
                        '../imp/feature_importance_lgbm.csv')

    if not debug:
        # save out of fold prediction
        train_df.loc[:,'demand'] = oof_preds
        train_df = train_df.reset_index()
        train_df[['id', 'demand']].to_csv(oof_file_name, index=False)

        # reshape prediction for submit
        test_df.loc[:,'demand'] = sub_preds
        test_df = test_df.reset_index()
        preds = test_df[['id','d','demand']].reset_index()
        preds = preds.pivot(index='id', columns='d', values='demand').reset_index()

        # split test1 / test2
        preds1 = preds[['id']+COLS_TEST1]
        preds2 = preds[['id']+COLS_TEST2]

        # change column names
        preds1.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
        preds2.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]

        # replace test2 id
        preds2['id']= preds2['id'].str.replace('_validation','_evaluation')

        # merge
        preds = preds1.append(preds2)

        # save csv
        preds.to_csv(submission_file_name, index=False)

        # submission by API
        submit(submission_file_name, comment='model201 cv: %.6f' % full_rmse)

def main(debug=False):
    with timer("Load Datasets"):
        # load feathers
        files = sorted(glob('../feats/*.feather'))
        df = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)], axis=1)

        # use selected features
        df = df[configs['features']]

        # set id as index
        df.set_index('id', inplace=True)

        # sort by date
        df.sort_values('date',inplace=True)

        df = df[df['date']>'2014-04-25']

        # split train & test
        train_df = df[df['date']<'2016-04-25']
        test_df = df[df['date']>='2016-04-25']
        del df
        gc.collect()
    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(train_df, test_df, num_folds=NUM_FOLDS, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission_lgbm.csv"
    oof_file_name = "../output/oof_lgbm.csv"
    configs = json.load(open('../configs/201_lgbm.json'))
    with timer("Full model run"):
        main(debug=False)