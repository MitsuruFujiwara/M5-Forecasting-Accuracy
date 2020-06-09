
import gc
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings

from glob import glob
from tqdm import tqdm

from utils import submit, WRMSSEEvaluator
from utils import FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2
from utils_lag import make_lags

#==============================================================================
# weekly prediction
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
    sub_28days = pd.read_csv("../output/submission_lgbm_28days.csv")
    sub_21days = pd.read_csv("../output/submission_lgbm_21days.csv")
    sub_14days = pd.read_csv("../output/submission_lgbm_14days.csv")
    sub_7days  = pd.read_csv("../output/submission_lgbm_7days.csv")

    # load out of fold files
    oof_28days = pd.read_csv("../output/oof_lgbm_28days.csv")
    oof_21days = pd.read_csv("../output/oof_lgbm_21days.csv")
    oof_14days = pd.read_csv("../output/oof_lgbm_14days.csv")
    oof_7days  = pd.read_csv("../output/oof_lgbm_7days.csv")

    # to pivot
    oof_28days = oof_28days.pivot(index='id', columns='d', values='demand').reset_index()
    oof_21days = oof_21days.pivot(index='id', columns='d', values='demand').reset_index()
    oof_14days = oof_14days.pivot(index='id', columns='d', values='demand').reset_index()
    oof_7days  = oof_7days.pivot(index='id', columns='d', values='demand').reset_index()

    # TODO: calc WRMSSE score

    # split columns
    col_28days = [f'F{i+1}' for i in range(21,28)]
    col_21days = [f'F{i+1}' for i in range(14,21)]
    col_14days = [f'F{i+1}' for i in range(7,14)]
    col_7days = [f'F{i+1}' for i in range(0,7)]

    # merge
    sub = sub_7days[['id']+col_7days].merge(sub_14days[['id']+col_14days],on='id',how='left')
    sub = sub.merge(sub_21days[['id']+col_21days],on='id',how='left')
    sub = sub.merge(sub_28days[['id']+col_28days],on='id',how='left')

    # postprocesssing
    cols_f = [f'F{i}' for i in range(1,29)]
    sub.loc[:,cols_f] = sub[cols_f].where(sub[cols_f]>0,0)

    # save csv
    sub.to_csv(submission_file_name, index=False)

    # TODO: calc oof score

    # submission by API
    submit(submission_file_name, comment='model401 weekly prediction')

if __name__ == '__main__':
    submission_file_name = "../output/submission_weekly.csv"
    main()
