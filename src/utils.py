
import feather
import gc
import json
import os
import pandas as pd
import numpy as np
import requests
import pickle

from glob import glob
from multiprocessing import Pool, cpu_count
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from time import time, sleep
from tqdm import tqdm

NUM_FOLDS = 5

FEATS_EXCLUDED = ['demand','index','id','d','date','is_test','wm_yr_wk','d_numeric','key_kf','is_zero']

# ref:https://www.kaggle.com/kailex/m5-forecaster-0-57330/comments
CAT_COLS = ['state_id','dept_id','cat_id','wday','day','week','month',
            'quarter', 'year', 'is_weekend','snap_CA', 'snap_TX', 'snap_WI']

COMPETITION_NAME = 'm5-forecasting-accuracy'

COLS_TEST1 = [f'd_{i}' for i in range(1914,1942)]
COLS_TEST2 = [f'd_{i}' for i in range(1942,1970)]

# to feather
def to_feature(df, path):
    if df.columns.duplicated().sum()>0:
        raise Exception('duplicated!: {}'.format(df.columns[df.columns.duplicated()]))
    df.reset_index(inplace=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather('{}_{}.feather'.format(path,c))
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

# Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

# stop GCP instance
def stop_instance():
    """
    You need to login first.
    >> gcloud auth login
    """
    send_line('stop instance')
#    os.system(f'gcloud compute instances stop {os.uname()[1]} --zone us-east1-b')
    os.system(f'gcloud compute instances stop {os.uname()[1]}\n\n')

# define cost and eval functions
def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * 1.15)
    hess = np.where(residual < 0, 2, 2 * 1.15)
    return grad, hess

def custom_asymmetric_valid(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2) , (residual ** 2) * 1.15)
    return "custom_asymmetric_eval", np.mean(loss), False

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
