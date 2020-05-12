
import feather
import gc
import json
import os
import pandas as pd
import numpy as np
import requests
import pickle

from multiprocessing import Pool, cpu_count
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from time import time, sleep
from tqdm import tqdm
from typing import Union

NUM_FOLDS = 5

FEATS_EXCLUDED = ['demand','index','id','d','date','is_test','wm_yr_wk','d_numeric','key_kf']

# ref:https://www.kaggle.com/kailex/m5-forecaster-0-57330/comments
CAT_COLS = ['state_id','dept_id','cat_id','wday','day','week','month',
            'quarter', 'year', 'is_weekend','snap_CA', 'snap_TX', 'snap_WI']

COMPETITION_NAME = 'm5-forecasting-accuracy'

COLS_TEST1 = [f'd_{i}' for i in range(1914,1942)]
COLS_TEST2 = [f'd_{i}' for i in range(1942,1970)]

DAYS_PRED = 28

# to feather
def to_feature(df, path):
    if df.columns.duplicated().sum()>0:
        raise Exception('duplicated!: {}'.format(df.columns[df.columns.duplicated()]))
    df.reset_index(inplace=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather('{}/{}.feather'.format(path,c))
    return

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# remove correlated variables
def removeCorrelatedVariables(data, threshold):
    print('Removing Correlated Variables...')
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    col_drop = [column for column in upper.columns if any(upper[column] > threshold) & ('target' not in column)]
    return col_drop

# remove missing variables
def removeMissingVariables(data, threshold):
    print('Removing Missing Variables...')
    missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    col_missing = missing.index[missing > threshold]
    col_missing = [column for column in col_missing if 'target' not in column]
    return col_missing

# LINE Notify
def line_notify(message):
    f = open('../input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    print(message)

# API submission https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/py/utils.py
def submit(file_path, comment='from API'):
    os.system('kaggle competitions submit -c {} -f {} -m "{}"'.format(COMPETITION_NAME,file_path,comment))
    sleep(360) # tekito~~~~
    tmp = os.popen('kaggle competitions submissions -c {} -v | head -n 2'.format(COMPETITION_NAME)).read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += '{}: {}\n'.format(i,j)
#        print(f'{i}: {j}') # TODO: comment out later?
    line_notify(message.rstrip())

# make dir
def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

# save pkl
def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close

# load pkl
def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out

# save multi-pkl files
def to_pickles(df, path, split_size=3):
    """
    path = '../output/mydf'
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    """
    print('shape: {}'.format(df.shape))

    gc.collect()
    mkdir_p(path)

    kf = KFold(n_splits=split_size, random_state=326)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(path+'/'+str(i)+'.pkl')
    return

# read multi-pkl files
def read_pickles(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([ pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*'))) ])
        else:
            print('reading {}'.format(path))
            df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(path+'/*')) ])
    else:
        df = pd.concat([ pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*'))) ])
    return df

# reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print('Reducing memory usage...')
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# save json
def to_json(data_dict, path):
    with open(path, 'w') as f:
        json.dump(data_dict, f, indent=4)

# stop GCP instance
def stop_instance():
    """
    You need to login first.
    >> gcloud auth login
    """
    send_line('stop instance')
#    os.system(f'gcloud compute instances stop {os.uname()[1]} --zone us-east1-b')
    os.system(f'gcloud compute instances stop {os.uname()[1]}\n\n')

# custom time series splitter
class CustomTimeSeriesSplitter(object):
    """
    split validation data
    fold 1 train: d_1 ~ d_1885, valid: d_1886 ~ d_1913
    fold 2 train: d_1 ~ d_1857, valid: d_1858 ~ d_1885
    fold 3 train: d_1 ~ d_1547, valid: d_1548 ~ d_1575

    public: d_1914 ~ d_1941
    private: d_1942 ~ d_1969
    """

    def __init__(self,end_train=1913,n_splits=3):
        self.end_train = end_train
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        # reset index
        X.reset_index(inplace=True)

        # get masks
        train_mask1 = (X['d_numeric']>=self.end_train-28-365*2)&(X['d_numeric']<=self.end_train-28)
        valid_mask1 = (X['d_numeric']>self.end_train-28)&(X['d_numeric']<=self.end_train)

        train_mask2 = (X['d_numeric']>=self.end_train-28*2-365*2)&(X['d_numeric']<=self.end_train-28*2)
        valid_mask2 = (X['d_numeric']>self.end_train-28*2)&(X['d_numeric']<=self.end_train-28)

        train_mask3 = (X['d_numeric']>=self.end_train-365-365*2)&(X['d_numeric']<=self.end_train-365)
        valid_mask3 = (X['d_numeric']>self.end_train-365)&(X['d_numeric']<=self.end_train-365+28)

        masks = [(train_mask1,valid_mask1),(train_mask2,valid_mask2),(train_mask3,valid_mask3)]

        for idx in range(self.n_splits):
            yield X[masks[idx][0]].index.values, X[masks[idx][1]].index.values

    def get_n_splits(self):
        return self.n_splits


# helper for making lag features
def make_lags(df):
    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm(range(1,15)):
        df[f'demand_lag_{i}'] = df_grouped.shift(DAYS_PRED+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,30,60,180]):
        df[f'demand_rolling_mean_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).mean())
        df[f'demand_rolling_std_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).std())

    del df_grouped
    gc.collect()

    # diff features
    df_grouped_diff = df[['id','demand']].groupby(['id'])['demand'].diff()
    print('Add lag features...')
    for i in tqdm(range(1,15)):
        df[f'demand_diff_lag_{i}'] = df_grouped_diff.shift(DAYS_PRED+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,30,60,180]):
        df[f'demand_diff_rolling_mean_{i}'] = df_grouped_diff.transform(lambda x: x.shift(DAYS_PRED).rolling(i).mean())
        df[f'demand_diff_rolling_std_{i}'] = df_grouped_diff.transform(lambda x: x.shift(DAYS_PRED).rolling(i).std())

    del df_grouped_diff
    gc.collect()

    return df


# function for evaluating WRMSSEE
# ref: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834
class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            setattr(self, f'lv{i + 1}_train_df', train_df.groupby(group_id)[train_target_columns].sum())
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        train_y = getattr(self, f'lv{lv}_train_df')
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = ((train_y.iloc[:, 1:].values - train_y.iloc[:, :-1].values) ** 2).mean(axis=1)
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        return np.mean(all_scores)
