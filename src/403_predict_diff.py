
import gc
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings

from glob import glob
from tqdm import tqdm

from utils import submit
from utils import FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2
from utils_lag import make_lags

#==============================================================================
# make prediction with diff data
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
    sub = pd.read_csv('../output/submission_lgbm_diff.csv')

    # load train data
    df = pd.read_csv('../input/sales_train_validation.csv')

    # to cumsum
    sub['F1'] += df['d_1913']
    sub.loc[:,'F1'] = sub['F1'].where(sub['F1']>0,0)

    for i in range(2,29):
        sub[f'F{i}'] += sub[f'F{i-1}']
        sub.loc[:,f'F{i}'] = sub[f'F{i}'].where(sub[f'F{i}']>0,0)

    # save csv
    sub.to_csv(submission_file_name, index=False)

    # submission by API
    submit(submission_file_name, comment='model401 weekly prediction')

if __name__ == '__main__':
    submission_file_name = '../output/submission_lgbm_diff.csv'
    main()
