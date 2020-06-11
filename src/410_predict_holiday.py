
import gc
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
import warnings

from glob import glob
from tqdm import tqdm

from utils import submit, line_notify
from utils import FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2
from utils_lag import make_lags
from utils_score import calc_score_cv

#==============================================================================
# prediction by holiday
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
    print('load files...')
    sub_weekday = pd.read_csv('../output/submission_lgbm_weekday.csv')
    sub_holiday = pd.read_csv('../output/submission_lgbm_holiday.csv')

    # load oof files
    oof_weekday = pd.read_csv('../output/oof_lgbm_cv_weekday.csv')
    oof_holiday = pd.read_csv('../output/oof_lgbm_cv_holiday.csv')

    # merge
    sub = sub_weekday.append(sub_holiday)

    oof = oof_weekday.append(oof_holiday)

    del sub_weekday, sub_holiday, oof_weekday, oof_holiday
    gc.collect()

    # to pivot
    print('to pivot...')
    sub = sub.pivot(index='id', columns='d', values='demand').reset_index()
    oof = oof.pivot(index='id', columns='d', values='demand').reset_index()

    # split test1 / test2
    sub1 = oof[['id']+COLS_TEST1]
    sub2 = sub[['id']+COLS_TEST2]

    # change column names
    sub1.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
    sub2.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]

    # replace test1 id
    sub1['id']= sub1['id'].str.replace('_evaluation','_validation')

    # merge
    sub = sub1.append(sub2)

    # postprocesssing
    cols_f = [f'F{i}' for i in range(1,29)]
    cols_d = [c for c in oof.columns if 'd_' in c]

    sub.loc[:,cols_f] = sub[cols_f].where(sub[cols_f]>0,0)
    oof.loc[:,cols_d] = oof[cols_d].where(oof[cols_d]>0,0)

    # save csv
    sub.to_csv(submission_file_name, index=False)
    oof.to_csv(oof_file_name, index=False)

    # calc out of fold WRMSSE score
    print('calc oof cv scores...')
    scores = calc_score_cv(oof)
    score = np.mean(scores)
    print(f'scores: {scores}')

    # submission by API
    submit(submission_file_name, comment='model410 cv: %.6f' % score)

    # LINE notify
    line_notify('{} done. WRMSSE:{}'.format(sys.argv[0],round(score,6)))

if __name__ == '__main__':
    submission_file_name = '../output/submission_holiday.csv'
    oof_file_name = '../output/oof_holiday_pivot.csv'
    main()
