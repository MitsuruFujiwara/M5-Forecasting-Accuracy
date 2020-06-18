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
from utils import COLS_TEST1, COLS_TEST2
from utils_score import calc_score_cv

#==============================================================================
# blending with ridge regression
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
#    sub1 = pd.read_csv('../output/submission_lgbm_28days.csv') # 0.61462
#    sub2 = pd.read_csv('../output/submission_cat_id.csv') # 0.62038
    sub3 = pd.read_csv('../output/submission_cat_id_group_k_fold.csv') # 0.59688
    sub4 = pd.read_csv('../output/submission_weekly.csv') # 0.59736
    sub5 = pd.read_csv('../output/submission_lgbm_group_k_fold_28days_pivot.csv') # 0.64584
    sub6 = pd.read_csv('../output/submission_weekly_group_k_fold.csv') # 0.62186
#    sub7 = pd.read_csv('../output/submission_holiday.csv') # 0.60812
#    sub8 = pd.read_csv('../output/submission_holiday_group_k_fold.csv') # 0.60812

    # load oof files
#    oof1 = pd.read_csv('../output/oof_lgbm_cv_28days_pivot.csv')
#    oof2 = pd.read_csv('../output/oof_cat_id.csv')
    oof3 = pd.read_csv('../output/oof_cat_id_group_k_fold.csv')
    oof4 = pd.read_csv('../output/oof_lgbm_direct_weekly_pivot.csv')
    oof5 = pd.read_csv('../output/oof_lgbm_group_k_fold_pivot.csv')
    oof6 = pd.read_csv('../output/oof_lgbm_weekly_group_k_fold_pivot.csv')
#    oof7 = pd.read_csv('../output/oof_holiday_pivot.csv')
#    oof8 = pd.read_csv('../output/oof_holiday_group_k_fold.csv')

    # load target data & sample submission
    df = pd.read_csv('../input/sales_train_evaluation.csv')
    sample_sub = pd.read_csv('../input/sample_submission.csv')

    # validation columns
    end_train=1941
    cols_valid1 = [f'd_{c}' for c in range(end_train-28+1,end_train+1)]
    cols_valid2 = [f'd_{c}' for c in range(end_train-28*2+1,end_train-28+1)]
    cols_valid3 = [f'd_{c}' for c in range(end_train-365+1,end_train-365+28+1)]

    # drop non-validation columns
    df = df[['id']+cols_valid3+cols_valid2+cols_valid1]
#    oof1 = oof1[['id']+cols_valid3+cols_valid2+cols_valid1]
#    oof2 = oof2[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof3 = oof3[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof4 = oof4[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof5 = oof5[['id']+cols_valid3+cols_valid2+cols_valid1]
    oof6 = oof6[['id']+cols_valid3+cols_valid2+cols_valid1]
#    oof7 = oof7[['id']+cols_valid3+cols_valid2+cols_valid1]
#    oof8 = oof8[['id']+cols_valid3+cols_valid2+cols_valid1]

    # reshape sub
    sample_sub = pd.melt(sample_sub,id_vars='id',var_name='d',value_name='demand')
#    sub1 = pd.melt(sub1,id_vars='id',var_name='d',value_name='sub1')
#    sub2 = pd.melt(sub2,id_vars='id',var_name='d',value_name='sub2')
    sub3 = pd.melt(sub3,id_vars='id',var_name='d',value_name='sub3')
    sub4 = pd.melt(sub4,id_vars='id',var_name='d',value_name='sub4')
    sub5 = pd.melt(sub5,id_vars='id',var_name='d',value_name='sub5')
    sub6 = pd.melt(sub6,id_vars='id',var_name='d',value_name='sub6')
#    sub7 = pd.melt(sub7,id_vars='id',var_name='d',value_name='sub7')
#    sub8 = pd.melt(sub8,id_vars='id',var_name='d',value_name='sub8')

    # aggregate sub
    sub = sample_sub.merge(sub3, on=['id','d'],how='left')
#    sub = sub.merge(sub2, on=['id','d'],how='left')
#    sub = sub.merge(sub3, on=['id','d'],how='left')
    sub = sub.merge(sub4, on=['id','d'],how='left')
    sub = sub.merge(sub5, on=['id','d'],how='left')
    sub = sub.merge(sub6, on=['id','d'],how='left')
#    sub = sub.merge(sub7, on=['id','d'],how='left')
#    sub = sub.merge(sub8, on=['id','d'],how='left')

    # reshape oof
    df = pd.melt(df,id_vars='id',var_name='d',value_name='demand')
#    oof1 = pd.melt(oof1,id_vars='id',var_name='d',value_name='oof1')
#    oof2 = pd.melt(oof2,id_vars='id',var_name='d',value_name='oof2')
    oof3 = pd.melt(oof3,id_vars='id',var_name='d',value_name='oof3')
    oof4 = pd.melt(oof4,id_vars='id',var_name='d',value_name='oof4')
    oof5 = pd.melt(oof5,id_vars='id',var_name='d',value_name='oof5')
    oof6 = pd.melt(oof6,id_vars='id',var_name='d',value_name='oof6')
#    oof7 = pd.melt(oof7,id_vars='id',var_name='d',value_name='oof7')
#    oof8 = pd.melt(oof8,id_vars='id',var_name='d',value_name='oof8')

    # aggregate oof
    df = df.merge(oof3, on=['id','d'],how='left')
#    df = df.merge(oof2, on=['id','d'],how='left')
#    df = df.merge(oof3, on=['id','d'],how='left')
    df = df.merge(oof4, on=['id','d'],how='left')
    df = df.merge(oof5, on=['id','d'],how='left')
    df = df.merge(oof6, on=['id','d'],how='left')
#    df = df.merge(oof7, on=['id','d'],how='left')
#    df = df.merge(oof8, on=['id','d'],how='left')

    # calc weights by ridge regression
    cols_oofs = ['oof3','oof4','oof5','oof6']
    reg = Ridge(alpha=1.0,fit_intercept=False,random_state=326)
    reg.fit(df[cols_oofs],df['demand'])

    print('weights: {}'.format(reg.coef_))

    # blending
    sub['demand'] = reg.coef_[0]*sub['sub3']+reg.coef_[1]*sub['sub4']+reg.coef_[2]*sub['sub5'] \
                  + reg.coef_[3]*sub['sub6']

    df['oof'] = reg.coef_[0]*df['oof3']+reg.coef_[1]*df['oof4']+reg.coef_[2]*df['oof5'] \
              + reg.coef_[3]*df['oof6']

    # postprocesssing
    sub.loc[:,'demand'] = sub['demand'].where(sub['demand']>0,0)
    df.loc[:,'oof'] = df['oof'].where(df['oof']>0,0)

    # to pivot
    sub = sub[['id','d','demand']].pivot(index='id', columns='d', values='demand').reset_index()
    oof = df[['id','d','oof']].pivot(index='id', columns='d', values='oof').reset_index()

    # split test1 / test2
    sub1 = oof[['id']+COLS_TEST1]
    sub2 = sub[['id']+['F' + str(d + 1) for d in range(28)]]

    # change column names
    sub1.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]

    # drop
    sub2[preds_valid.id.str.contains("validation")]

    # replace test1 id
    sub1['id']= sub1['id'].str.replace('_evaluation','_validation')

    # merge
    sub = sub1.append(sub2)

    # calc out of fold WRMSSE score
    print('calc oof cv scores...')
    scores = calc_score_cv(oof)
    score = np.mean(scores)
    print(f'scores: {scores}')

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
