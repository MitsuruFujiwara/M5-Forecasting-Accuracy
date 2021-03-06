
import feather
import gc
import json
import numpy as np
import pandas as pd
import sys
import warnings

from utils import loadpkl, to_feature, line_notify, to_json, read_pickles
from utils import removeCorrelatedVariables, removeMissingVariables, reduce_mem_usage
from utils_lag import make_lags

#===============================================================================
# aggregation (21days lag)
#===============================================================================

warnings.simplefilter(action='ignore')

def main():
    # load pkls
    df = read_pickles('../feats/sales')
    df_calendar = loadpkl('../feats/calendar.pkl')
    df_sell_prices = loadpkl('../feats/sell_prices.pkl')

    # merge
    df = df.merge(df_calendar, on='d',how='left')
    df = df.merge(df_sell_prices, on=['store_id','item_id','wm_yr_wk'],how='left')

    del df_calendar, df_sell_prices
    gc.collect()

    # drop pre-release rows
    df = df[df['wm_yr_wk']>=df['release']]

    # make lag features
    df = make_lags(df,21)

    # add categorical features
    df['item_id_store_id'] = df['item_id']+'_'+df['store_id']
    df['item_id_state_id'] = df['item_id']+'_'+df['state_id']
    df['dept_id_store_id'] = df['dept_id']+'_'+df['store_id']
    df['dept_id_state_id'] = df['dept_id']+'_'+df['state_id']

    # label encoding
    cols_string = ['item_id','dept_id','cat_id','store_id','state_id',
                   'item_id_store_id','item_id_state_id','dept_id_store_id',
                   'dept_id_state_id']
    for c in cols_string:
        df[c], _ = pd.factorize(df[c])
        df[c].replace(-1,np.nan,inplace=True)

    # add price features
    df_grouped = df[['id','sell_price']].groupby('id')['sell_price']
    df['shift_price_t1'] = df_grouped.transform(lambda x: x.shift(1))
    df['price_change_t1'] = (df['shift_price_t1'] - df['sell_price']) / (df['shift_price_t1'])
    df['rolling_price_max_t365'] = df_grouped.transform(lambda x: x.shift(1).rolling(365).max())
    df['price_change_t365'] = (df['rolling_price_max_t365'] - df['sell_price']) / (df['rolling_price_max_t365'])
    df['rolling_price_std_t7'] = df_grouped.transform(lambda x: x.rolling(7).std())
    df['rolling_price_std_t30'] = df_grouped.transform(lambda x: x.rolling(30).std())

    # features release date
    df['release'] = df['release'] - df['release'].min()

    # price momentum by month & year
    df['price_momentum_m'] = df['sell_price']/df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
    df['price_momentum_y'] = df['sell_price']/df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

    # days for CustomTimeSeriesSplitter
    df['d_numeric'] = df['d'].apply(lambda x: str(x)[2:]).astype(int)

    # reduce memory usage
    df = reduce_mem_usage(df)

    # save as feather
    to_feature(df, '../feats/f102')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/102_all_features_21days.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
