
import gc
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
import warnings

from glob import glob
from tqdm import tqdm

from utils import submit, WRMSSEEvaluator, line_notify
from utils import FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2
from utils_lag import make_lags

#==============================================================================
# direct 28days prediction
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
    print('load files...')
    sub = pd.read_csv(submission_file_name )

    # load out of fold files
    oof = pd.read_csv(oof_file_name)

    # load files
    df = pd.read_csv('../input/sales_train_evaluation.csv')
    calendar = pd.read_csv('../input/calendar.csv')
    prices = pd.read_csv('../input/sell_prices.csv')
    sample_sub = pd.read_csv("../input/sample_submission.csv")

    # order for sort
    sample_sub["order"] = range(sample_sub.shape[0])

    # to pivot
    print('to pivot...')
    oof = oof.pivot(index='id', columns='d', values='demand').reset_index()

    # postprocesssing
    cols_f = [f'F{i}' for i in range(1,29)]
    cols_d = [c for c in oof.columns if 'd_' in c]
    sub.loc[:,cols_f] = sub[cols_f].where(sub[cols_f]>0,0)
    oof.loc[:,cols_d] = oof[cols_d].where(oof[cols_d]>0,0)

    # save csv
    sub.to_csv(submission_file_name, index=False)
    oof.to_csv(oof_file_name, index=False)

    # calc out of fold WRMSSE score
    # get dates for validation
    cols_valid = []
    for c in cols_d:
        if (oof[c]>0).sum() != 0:
            cols_valid.append(c)

    # split train & valid
    df_train = df.iloc[:, :-28]
    df_valid = df[cols_valid]

    del df
    gc.collect()

    # sort oof
    oof = oof.merge(sample_sub[["id", "order"]], on = "id")
    oof.sort_values("order",inplace=True)
    oof.drop(["id", "order"],axis=1,inplace=True)
    oof.reset_index(drop=True, inplace=True)

    # calc score
    evaluator = WRMSSEEvaluator(df_train, df_valid, calendar, prices)
    groups, scores = evaluator.score(oof[cols_valid])
    score = np.mean(scores)

    # submission by API
    submit(submission_file_name, comment='model401 cv: %.6f' % score)

    # LINE notify
    line_notify('{} done. WRMSSE:{}'.format(sys.argv[0],score))

if __name__ == '__main__':
    submission_file_name = '../output/submission_lgbm_28days.csv'
    oof_file_name = '../output/oof_lgbm_cv_28days.csv'
    main()
