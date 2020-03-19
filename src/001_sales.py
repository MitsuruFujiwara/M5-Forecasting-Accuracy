
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from tqdm import tqdm

from utils import save2pkl, line_notify
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

    # shifted features
    for diff in [28, 365]:
        df[f'demand_shift_{diff}'] = df.groupby('id').shift(diff)['demand']

    # save pkl
    save2pkl('../feats/sales.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main(is_eval=False)