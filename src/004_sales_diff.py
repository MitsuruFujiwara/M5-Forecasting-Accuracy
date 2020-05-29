
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from tqdm import tqdm

from utils import save2pkl, line_notify, reduce_mem_usage, to_pickles
from utils import COLS_TEST1, COLS_TEST2

#===============================================================================
# preprocess sales diff
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

    # drop christmas data
    days_christmas = ['d_1062','d_1427','d_1792']
    df.drop(days_christmas,axis=1,inplace=True)

    # to diff
    print('to diff...')
    cols_d = [c for c in df.columns if 'd_' in c]
    df.loc[:,cols_d] = df[cols_d].diff(axis=1)

    # drop old data (~2012/12/31)
    print('drop old data...')
    days_old = [f'd_{i}' for i in range(1,705)]
    df.drop(days_old,axis=1,inplace=True)

    # reduce memory usage
    df = reduce_mem_usage(df)

    # melt sales data
    print('Melting sales data...')
    id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id']
    df = pd.melt(df,id_vars=id_vars,var_name='d',value_name='demand')

    print('Melted sales train validation has {} rows and {} columns'.format(df.shape[0], df.shape[1]))

    # add numeric date
    df['d_numeric'] = df['d'].apply(lambda x: int(x[2:]))

    # save pkl
    to_pickles(df, '../feats/sales_diff', split_size=3)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main(is_eval=False)
