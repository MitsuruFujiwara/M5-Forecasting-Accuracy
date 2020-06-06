
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
from glob import glob
from tqdm import tqdm

from utils import line_notify, to_json, rmse, save2pkl, submit
from utils import NUM_FOLDS, FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2, CAT_COLS
from utils import CustomTimeSeriesSplitter, custom_asymmetric_train, custom_asymmetric_valid

#==============================================================================
# Train LightGBM with custom cv (foods)
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
def kfold_lightgbm(train_df, test_df, num_folds):
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))

    # Cross validation
    folds = CustomTimeSeriesSplitter(end_train=1941)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    valid_idxs=[]
    avg_best_iteration = 0 # average of best iteration

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df)):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['demand'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['demand'].iloc[valid_idx]

        # TODO: target encoding

        # save validation indexes
        valid_idxs += list(valid_idx)

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        params ={
#                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'learning_rate': 0.1,
                'bagging_fraction': 0.85,
                'bagging_freq': 1,
                'colsample_bytree': 0.85,
                'colsample_bynode': 0.85,
                'min_data_per_leaf': 25,
                'num_leaves': 200,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                'verbose': -1,
                'seed':326,
                'bagging_seed':326,
                'drop_seed':326,
                'num_threads':-1
                }

        # train model
        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        fobj = custom_asymmetric_train,
                        feval = custom_asymmetric_valid,
                        early_stopping_rounds=200,
                        verbose_eval=10
                        )

        # save model
        reg.save_model(f'../output/lgbm_foods_{n_fold}.txt')

        # save predictions
        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        # save best iteration
        avg_best_iteration += reg.best_iteration / folds.n_splits

        # save feature importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # display importances
    display_importances(feature_importance_df,
                        '../imp/lgbm_importances_cv_foods.png',
                        '../imp/feature_importance_lgbm_cv_foods.csv')

    # Full RMSE score and LINE Notify
    full_rmse = rmse(train_df['demand'][valid_idxs], oof_preds[valid_idxs])
    line_notify('Full RMSE score %.6f' % full_rmse)

    # save out of fold prediction
    train_df.loc[:,'demand'] = oof_preds
    train_df[['id','d','demand']].to_csv(oof_file_name, index=False)

    # save number of best iteration
    configs['num_boost_round'] = int(avg_best_iteration)
    to_json(configs, '../configs/306_train_foods.json')

    # LINE notify
    line_notify('{} done. best iteration:{}'.format(sys.argv[0],int(avg_best_iteration)))

def main(is_eval=False):
    with timer("Load Datasets"):
        # load feathers
        files = sorted(glob('../feats/f106_*.feather'))
        df = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)], axis=1)

        # use selected features
        df = df[configs['features']]

        # drop old data
        df = df[df['date']>'2012-12-31']

        #=======================================================================
        # 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
        # 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
        # 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)
        #=======================================================================

        if is_eval:
            train_df = df[df['date']<'2016-05-23']
            test_df = df[df['date']>='2016-05-23']
        else:
            train_df = df[df['date']<'2016-04-25']
            test_df = df[df['date']>='2016-04-25']

        del df
        gc.collect()

    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(train_df, test_df, num_folds=NUM_FOLDS)

if __name__ == "__main__":
    oof_file_name = "../output/oof_lgbm_cv_foods.csv"
    configs = json.load(open('../configs/207_cv_foods.json'))
    with timer("Full model run"):
        main(is_eval=True)
