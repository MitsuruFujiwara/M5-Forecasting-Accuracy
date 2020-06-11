import gc

import json
import numpy as np
import pandas as pd
import sys
import warnings

from glob import glob
from sklearn.linear_model import Ridge
from tqdm import tqdm

from utils import submit, line_notify
from utils_score import calc_score_cv

#==============================================================================
# blending with ridge regression
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
#    sub1 = pd.read_csv('../output/submission_lgbm_28days.csv',index_col=0)
    sub1 = pd.read_csv('../output/submission_weekly_group_k_fold.csv',index_col=0)
    sub2 = pd.read_csv('../output/submission_cat_id.csv',index_col=0)
    sub3 = pd.read_csv('../output/submission_cat_id_group_k_fold.csv',index_col=0)
    sub4 = pd.read_csv('../output/submission_weekly.csv',index_col=0)
    oof_file_name_pivot = '../output/oof_lgbm_weekly_group_k_fold_pivot.csv'

    # load oof files
#    oof1 = pd.read_csv('../output/oof_lgbm_cv_28days.csv')
    oof1 = pd.read_csv('../output/oof_lgbm_weekly_group_k_fold_pivot.csv')
    oof2 = pd.read_csv('../output/oof_cat_id.csv')
    oof3 = pd.read_csv('../output/oof_cat_id_group_k_fold.csv')
    oof4 = pd.read_csv('../output/oof_lgbm_direct_weekly_pivot.csv')

    # load target data
    df = pd.read_csv('../input/sales_train_evaluation.csv')

    # validation columns
    end_train=1941
    cols_valid1 = [f'd_{c}' for c in range(end_train-28+1,end_train+1)]
    cols_valid2 = [f'd_{c}' for c in range(end_train-28*2+1,end_train-28+1)]
    cols_valid3 = [f'd_{c}' for c in range(end_train-365+1,end_train-365+28+1)]

    # drop non-validation columns
    df = df[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof1 = oof1[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof2 = oof2[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof3 = oof3[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof4 = oof4[['id']+cols_valid3+cols_valid2+cols_valid1]

    # reshape
    df = pd.melt(df,id_vars='id',var_name='d',value_name='demand')
    oof1 = pd.melt(oof1,id_vars='id',var_name='d',value_name='oof1')
    oof2 = pd.melt(oof2,id_vars='id',var_name='d',value_name='oof2')
    oof3 = pd.melt(oof3,id_vars='id',var_name='d',value_name='oof3')
    oof4 = pd.melt(oof4,id_vars='id',var_name='d',value_name='oof4')

    # aggregate predictions
    df = df.merge(oof1, on=['id','d'],how='left')
    df = df.merge(oof2, on=['id','d'],how='left')
    df = df.merge(oof3, on=['id','d'],how='left')
    df = df.merge(oof4, on=['id','d'],how='left')

    # calc weights by ridge regression
    reg = Ridge(alpha=1.0,fit_intercept=False,random_state=326)
    reg.fit(df[['oof1','oof2','oof3','oof4']],df['demand'])

    print('weights: {}'.format(reg.coef_))

    # blending
    sub = reg.coef_[0]*sub1+reg.coef_[1]*sub2+reg.coef_[2]*sub3+reg.coef_[3]*sub4
    df['oof'] = reg.coef_[0]*df['oof1']+reg.coef_[1]*df['oof2']+reg.coef_[2]*df['oof3']+reg.coef_[3]*df['oof4']

    # postprocesssing
    cols_f = [f'F{i}' for i in range(1,29)]
    sub.loc[:,cols_f] = sub[cols_f].where(sub[cols_f]>0,0)
    df.loc[:,'oof'] = df['oof'].where(df['oof']>0,0)

    # to pivot
    oof = df[['id','d','oof']].pivot(index='id', columns='d', values='oof').reset_index()

    # calc out of fold WRMSSE score
    print('calc oof cv scores...')
    scores = calc_score_cv(oof)
    score = np.mean(scores)
    print(f'scores: {scores}')

    # reset index
    sub.reset_index(inplace=True)
    oof.reset_index(inplace=True)

    # save csv
    sub.to_csv(submission_file_name, index=False)
    oof.to_csv(oof_file_name_pivot, index=False)

    # submission by API
    submit(submission_file_name, comment='model501 cv: %.6f' % score)

    # LINE notify
    line_notify('{} done. WRMSSE:{}'.format(sys.argv[0],round(score,6)))

if __name__ == '__main__':
    submission_file_name = '../output/submission_blend.csv'
    oof_file_name_pivot = '../output/oof_blend_pivot.csv'
    main()
