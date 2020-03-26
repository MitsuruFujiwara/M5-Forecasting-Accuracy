
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from tqdm import tqdm

from utils import save2pkl, line_notify, reduce_mem_usage
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

    # reduce memory usage
    df = reduce_mem_usage(df)

    # date columns
    cols_date = [c for c in df.columns if 'd_' in c]

    # replace pre-sale product amout as nan
    df['pre_sale_flag'] = True

    print('Replacing pre-sale product amout as nan...')
    for c in tqdm(cols_date):
        df.loc[(df[c]>0)&(df['pre_sale_flag']),'pre_sale_flag'] = False
        df.loc[df['pre_sale_flag'],c] = -1

    # drop pre-sale flag
    df.drop('pre_sale_flag',axis=1,inplace=True)

    # melt sales data
    print('Melting sales data...')
    id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id']
    df = pd.melt(df,id_vars=id_vars,var_name='d',value_name='demand')

    print('Melted sales train validation has {} rows and {} columns'.format(df.shape[0], df.shape[1]))

    # drop pre-sales data
    print('Dropping pre-sales data...')
    df = df[df['demand']>=0]

    print('Add demand features...')
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    # shifted demand
    for diff in [0,1,2,365]:
        df[f'demand_shift_{diff}'] = df_grouped.shift(DAYS_PRED+diff)

    # rolling mean
    for size in [7, 30, 60, 90, 180 ,365]:
        df[f'demand_mean_{size}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(size).mean())

    # rolling std
    for size in [7, 30, 60, 90, 180 ,365]:
        df[f'demand_std_{size}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(size).std())

    # rolling skew
    for size in [7, 30, 60, 90, 180 ,365]:
        df[f'demand_skew_{size}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(size).skew())

    # rolling kurt
    for size in [7, 30, 60, 90, 180 ,365]:
        df[f'demand_kurt_{size}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(size).kurt())

    # rolling max
    for size in [7, 30, 60, 90, 180 ,365]:
        df[f'demand_max_{size}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(size).max())

    # rolling min
    for size in [7, 30, 60, 90, 180 ,365]:
        df[f'demand_min_{size}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(size).min())

    # save pkl
    save2pkl('../feats/sales.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main(is_eval=False)
