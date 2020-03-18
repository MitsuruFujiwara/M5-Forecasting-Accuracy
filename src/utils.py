
import json
import feather
import os
import pandas as pd
import numpy as np
import requests
import pickle

from multiprocessing import Pool, cpu_count
from sklearn.metrics import mean_squared_error
from time import time, sleep
from tqdm import tqdm

NUM_FOLDS = 5

FEATS_EXCLUDED = ['demand','index','id','d','date','is_test','wm_yr_wk']

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
    sleep(60) # tekito~~~~
    tmp = os.popen('kaggle competitions submissions -c {} -v | head -n 2'.format(COMPETITION_NAME)).read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += '{}: {}\n'.format(i,j)
#        print(f'{i}: {j}') # TODO: comment out later?
    line_notify(message.rstrip())

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

# ref: https://www.kaggle.com/harupy/m5-baseline
class CustomTimeSeriesSplitter:
    def __init__(self, n_splits=5, train_days=80, test_days=20, day_col="d"):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.day_col = day_col

    def split(self, X, y=None, groups=None):
        SEC_IN_DAY = 3600 * 24
        sec = (X[self.day_col] - X[self.day_col].iloc[0]) * SEC_IN_DAY
        duration = sec.max()

        train_sec = self.train_days * SEC_IN_DAY
        test_sec = self.test_days * SEC_IN_DAY
        total_sec = test_sec + train_sec

        if self.n_splits == 1:
            train_start = duration - total_sec
            train_end = train_start + train_sec

            train_mask = (sec >= train_start) & (sec < train_end)
            test_mask = sec >= train_end

            yield sec[train_mask].index.values, sec[test_mask].index.values

        else:
            # step = (duration - total_sec) / (self.n_splits - 1)
            step = DAYS_PRED * SEC_IN_DAY

            for idx in range(self.n_splits):
                # train_start = idx * step
                shift = (self.n_splits - (idx + 1)) * step
                train_start = duration - total_sec - shift
                train_end = train_start + train_sec
                test_end = train_end + test_sec

                train_mask = (sec > train_start) & (sec <= train_end)

                if idx == self.n_splits - 1:
                    test_mask = sec > train_end
                else:
                    test_mask = (sec > train_end) & (sec <= test_end)

                yield sec[train_mask].index.values, sec[test_mask].index.values

    def get_n_splits(self):
        return self.n_splits
