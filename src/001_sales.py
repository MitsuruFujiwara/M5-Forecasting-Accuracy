
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from tqdm import tqdm

from utils import save2pkl, line_notify, reduce_mem_usage, to_pickles
from utils import COLS_TEST1, COLS_TEST2, DAYS_PRED

#===============================================================================
# preprocess sales
#===============================================================================

warnings.simplefilter(action='ignore')

def main(is_eval=False):
    # load csv
    if is_eval:
        df = pd.read_csv('../input/sales_train_evaluation.csv')
    else:
        df = pd.read_csv('../input/sales_train_validation.csv')

    sub = pd.read_csv('../input/sample_submission.csv')

    # split test data
    sub['is_test1']=sub['id'].apply(lambda x: True if '_validation' in x else False)
    sub['is_test2']=sub['id'].apply(lambda x: True if '_evaluation' in x else False)

    test1 = sub[sub['is_test1']]
    test2 = sub[sub['is_test2']]

    del sub
    gc.collect()

    # drop flags
    test1.drop(['is_test1','is_test2'],axis=1,inplace=True)
    test2.drop(['is_test1','is_test2'],axis=1,inplace=True)

    # change column name
    test1.columns = ['id']+COLS_TEST1
    test2.columns = ['id']+COLS_TEST2

    # change id
    test2['id'] = test2['id'].str.replace('_evaluation','_validation')

    # merge
    df = df.merge(test1,on='id',how='left')
    df = df.merge(test2,on='id',how='left')

    del test1, test2
    gc.collect()

    # date columns
    cols_date = [c for c in df.columns if 'd_' in c]

    # melt sales data
    print('Melting sales data...')
    id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id']
    df = pd.melt(df,id_vars=id_vars,var_name='d',value_name='demand')

    print('Melted sales train validation has {} rows and {} columns'.format(df.shape[0], df.shape[1]))

    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm(range(1,15)):
        df[f'demand_lag_{i}'] = df_grouped.shift(DAYS_PRED+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,30,60,180]):
        df[f'demand_rolling_mean_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).mean())
        df[f'demand_rolling_std_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).std())

    del df_grouped
    gc.collect()

    # diff features
    df_grouped_diff = df[['id','demand']].groupby(['id'])['demand'].diff()
    print('Add lag features...')
    for i in tqdm(range(1,15)):
        df[f'demand_diff_lag_{i}'] = df_grouped_diff.shift(DAYS_PRED+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,30,60,180]):
        df[f'demand_diff_rolling_mean_{i}'] = df_grouped_diff.transform(lambda x: x.shift(DAYS_PRED).rolling(i).mean())
        df[f'demand_diff_rolling_std_{i}'] = df_grouped_diff.transform(lambda x: x.shift(DAYS_PRED).rolling(i).std())

    del df_grouped_diff
    gc.collect()

    # add numeric date
    df['d_numeric'] = df['d'].apply(lambda x: int(x[2:]))

    # drop old data (~2012/12/31)
    df = df[df['d_numeric']>=704]

    # reduce memory usage
    df = reduce_mem_usage(df)

    # save pkl
    save2pkl('../feats/sales.pkl', df)
#    to_pickles(df, '../feats/sales', split_size=3)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main(is_eval=False)
