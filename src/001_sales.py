
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from tqdm import tqdm

from utils import save2pkl, line_notify

#===============================================================================
# preprocess sales
#===============================================================================

warnings.simplefilter(action='ignore')

def main(is_eval=False):
    # load csv
    df = pd.read_csv('../input/sales_train_validation.csv')

    # date columns
    cols_date = [c for c in df.columns if 'd_' in c]

    # replace pre-sale product amout as nan
    df['pre_sale_flag'] = True

    print('replace pre-sale product amout as nan...')
    for c in tqdm(cols_date):
        df.loc[(df[c]>0)&(df['pre_sale_flag']),'pre_sale_flag'] = False
        df.loc[df['pre_sale_flag'],c] = -1

    # drop pre-sale flag
    df.drop('pre_sale_flag',axis=1,inplace=True)

    # melt sales data
    id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id']
    df = pd.melt(df,id_vars=id_vars,var_name='d',value_name='demand')

    # drop pre-sales data
    df = df[df['demand']>=0]

    # save pkl
    save2pkl('../feats/sales.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))


if __name__ == '__main__':
    main(is_eval=False)
