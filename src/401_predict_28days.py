
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
# direct 28days prediction
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
    print('load files...')
    sub = pd.read_csv(submission_file_name )

    # load out of fold files
    oof = pd.read_csv(oof_file_name)

    # to pivot
    print('to pivot...')
    oof = oof.pivot(index='id', columns='d', values='demand').reset_index()

    # fill na
    oof.fillna(0,inplace=True)

    # postprocesssing
    cols_f = [f'F{i}' for i in range(1,29)]
    cols_d = [c for c in oof.columns if 'd_' in c]
    sub.loc[:,cols_f] = sub[cols_f].where(sub[cols_f]>0,0)
    oof.loc[:,cols_d] = oof[cols_d].where(oof[cols_d]>0,0)

    # save csv
    sub.to_csv(submission_file_name, index=False)
    oof.to_csv(oof_file_name_pivot, index=False)

    # calc out of fold WRMSSE score
    print('calc oof cv scores...')
    scores = calc_score_cv(oof)
    score = np.mean(scores)
    print(f'scores: {scores}')

    # submission by API
#    submit(submission_file_name, comment='model401 cv: %.6f' % score)

    # LINE notify
    line_notify('{} done. WRMSSE:{}'.format(sys.argv[0],round(score,6)))

if __name__ == '__main__':
    submission_file_name = '../output/submission_lgbm_28days.csv'
    oof_file_name = '../output/oof_lgbm_cv_28days.csv'
    oof_file_name_pivot = '../output/oof_lgbm_cv_28days_pivot.csv'
    main()
