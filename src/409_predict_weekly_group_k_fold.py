
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
from utils_score import calc_score_cv

#==============================================================================
# weekly prediction with group k-fold
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    print('load files...')
    # load submission files
    sub_28days = pd.read_csv('../output/submission_lgbm_group_k_fold_28days.csv')
    sub_21days = pd.read_csv('../output/submission_lgbm_group_k_fold_21days.csv')
    sub_14days = pd.read_csv('../output/submission_lgbm_group_k_fold_14days.csv')
    sub_7days  = pd.read_csv('../output/submission_lgbm_group_k_fold_7days.csv')

    # load out of fold files
    oof_28days = pd.read_csv('../output/oof_lgbm_group_k_fold_28days.csv')
    oof_21days = pd.read_csv('../output/oof_lgbm_group_k_fold_21days.csv')
    oof_14days = pd.read_csv('../output/oof_lgbm_group_k_fold_14days.csv')
    oof_7days  = pd.read_csv('../output/oof_lgbm_group_k_fold_7days.csv')

    # to pivot
    print('to pivot...')
    sub_28days = sub_28days.pivot(index='id', columns='d', values='demand').reset_index()
    sub_21days = sub_21days.pivot(index='id', columns='d', values='demand').reset_index()
    sub_14days = sub_14days.pivot(index='id', columns='d', values='demand').reset_index()
    sub_7days  = sub_7days.pivot(index='id', columns='d', values='demand').reset_index()

    oof_28days = oof_28days.pivot(index='id', columns='d', values='demand').reset_index()
    oof_21days = oof_21days.pivot(index='id', columns='d', values='demand').reset_index()
    oof_14days = oof_14days.pivot(index='id', columns='d', values='demand').reset_index()
    oof_7days  = oof_7days.pivot(index='id', columns='d', values='demand').reset_index()

    # change columns name
    sub_28days.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
    sub_21days.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
    sub_14days.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
    sub_7days.columns  = ['id'] + ['F' + str(d + 1) for d in range(28)]

    # validation columns
    valid_col_28days_fold1 = [f'd_{i+1}' for i in range(1913+21,1913+28)]
    valid_col_21days_fold1 = [f'd_{i+1}' for i in range(1913+14,1913+21)]
    valid_col_14days_fold1 = [f'd_{i+1}' for i in range(1913+7,1913+14)]
    valid_col_7days_fold1  = [f'd_{i+1}' for i in range(1913,1913+7)]

    valid_col_28days_fold2 = [f'd_{i+1}' for i in range(1885+21,1885+28)]
    valid_col_21days_fold2 = [f'd_{i+1}' for i in range(1885+14,1885+21)]
    valid_col_14days_fold2 = [f'd_{i+1}' for i in range(1885+7,1885+14)]
    valid_col_7days_fold2  = [f'd_{i+1}' for i in range(1885,1885+7)]

    valid_col_28days_fold3 = [f'd_{i+1}' for i in range(1576+21,1576+28)]
    valid_col_21days_fold3 = [f'd_{i+1}' for i in range(1576+14,1576+21)]
    valid_col_14days_fold3 = [f'd_{i+1}' for i in range(1576+7,1576+14)]
    valid_col_7days_fold3  = [f'd_{i+1}' for i in range(1576,1576+7)]

    # merge oof files
    oof = oof_28days[['id']+valid_col_28days_fold1].merge(oof_28days[['id']+valid_col_28days_fold2],on='id',how='left')
    oof = oof.merge(oof_28days[['id']+valid_col_28days_fold3],on='id',how='left')

    oof = oof.merge(oof_21days[['id']+valid_col_21days_fold1],on='id',how='left')
    oof = oof.merge(oof_21days[['id']+valid_col_21days_fold2],on='id',how='left')
    oof = oof.merge(oof_21days[['id']+valid_col_21days_fold3],on='id',how='left')

    oof = oof.merge(oof_14days[['id']+valid_col_14days_fold1],on='id',how='left')
    oof = oof.merge(oof_14days[['id']+valid_col_14days_fold2],on='id',how='left')
    oof = oof.merge(oof_14days[['id']+valid_col_14days_fold3],on='id',how='left')

    oof = oof.merge(oof_7days[['id']+valid_col_7days_fold1],on='id',how='left')
    oof = oof.merge(oof_7days[['id']+valid_col_7days_fold2],on='id',how='left')
    oof = oof.merge(oof_7days[['id']+valid_col_7days_fold3],on='id',how='left')

    # split columns
    col_28days = [f'F{i+1}' for i in range(21,28)]
    col_21days = [f'F{i+1}' for i in range(14,21)]
    col_14days = [f'F{i+1}' for i in range(7,14)]
    col_7days = [f'F{i+1}' for i in range(0,7)]

    # merge
    sub = sub_7days[['id']+col_7days].merge(sub_14days[['id']+col_14days],on='id',how='left')
    sub = sub.merge(sub_21days[['id']+col_21days],on='id',how='left')
    sub = sub.merge(sub_28days[['id']+col_28days],on='id',how='left')

    # split test1 / test2
    sub1 = oof[['id']+COLS_TEST1]
    sub2 = sub[['id']+['F' + str(d + 1) for d in range(28)]]

    # change column names
    sub1.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]

    # replace test1 id
    sub1['id']= sub1['id'].str.replace('_evaluation','_validation')

    # merge
    sub = sub1.append(sub2)

    # postprocesssing
    cols_f = [f'F{i}' for i in range(1,29)]
    cols_d = [c for c in oof.columns if 'd_' in c]
    sub.loc[:,cols_f] = sub[cols_f].where(sub[cols_f]>0,0)
    oof.loc[:,cols_d] = oof[cols_d].where(oof[cols_d]>0,0)

    # calc out of fold WRMSSE score
    print('calc oof cv scores...')
    scores = calc_score_cv(oof)
    score = np.mean(scores)
    print(f'scores: {scores}')

    # save csv
    sub.to_csv(submission_file_name, index=False)
    oof.to_csv(oof_file_name_pivot, index=False)

    # submission by API
#    submit(submission_file_name, comment='model409 cv: %.6f' % score)

    # LINE notify
    line_notify('{} done. WRMSSE:{}'.format(sys.argv[0],round(score,6)))

if __name__ == '__main__':
    submission_file_name = '../output/submission_weekly_group_k_fold.csv'
    oof_file_name_pivot = '../output/oof_lgbm_weekly_group_k_fold_pivot.csv'
    main()
