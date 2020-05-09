
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from utils import save2pkl, line_notify

#===============================================================================
# preprocess sell prices
#===============================================================================

warnings.simplefilter(action='ignore')

def main(is_eval=False):
    # load csv
    df = pd.read_csv('../input/sell_prices.csv')

    # release week ref https://www.kaggle.com/kyakovlev/m5-simple-fe
    release_df = df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id','item_id','release']

    # merge release week
    df = df.merge(release_df, on=['store_id','item_id'],how='left')

    # basic aggregations
    df['price_max'] = df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    df['price_min'] = df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    df['price_std'] = df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    df['price_mean'] = df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

    # normalized price
    df['price_norm'] = df['sell_price']/df['price_max']

    # label encoding
    df['price_nunique'] = df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    df['item_nunique'] = df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

    # momentum
    df['price_momentum'] = df['sell_price']/df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))

    # save pkl
    save2pkl('../feats/sell_prices.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
