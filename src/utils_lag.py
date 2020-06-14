
import gc
import numpy as np
import pandas as pd

from tqdm import tqdm

#==============================================================================
# utils for lag features
#==============================================================================

# helper for making lag features
def make_lags(df,days=28):
    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm([0,7,14,21,28]):
        df[f'demand_lag_{days}_{i}'] = df_grouped.shift(days+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,21,28]):
        df[f'demand_rolling_mean_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).mean())
        df[f'demand_rolling_std_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).std())
        df[f'demand_rolling_max_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).max())
        df[f'demand_rolling_min_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).min())
        df[f'demand_rolling_skew_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).skew())
        df[f'demand_rolling_kurt_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).kurt())

    del df_grouped
    gc.collect()

    # is zero ratio
    df_grouped_is_zero = df[['id','is_zero']].groupby(['id'])['is_zero']

    print('Add is zero ratio...')
    for i in tqdm([7,14,21,28]):
        df[f'iz_zero_ratio_{days}_{i}'] = df_grouped_is_zero.transform(lambda x: x.shift(days).rolling(i).mean())

    # TODO: days after last sales

    del df_grouped_is_zero
    gc.collect()

    return df

# target encoding
def target_encoding(train_x,valid_x,train_y):
    cols_id = ['item_id','cat_id','dept_id','store_id','state_id']
    enc_cols = []
    print('target encoding...')
    for c in tqdm(cols_id):
        tmp_df = pd.DataFrame({c:train_x[c],'target':train_y})
        target_mean = tmp_df.groupby(c)['target'].mean()
        train_x[f'enc_{c}_mean'] = train_x[c].map(target_mean)
        valid_x[f'enc_{c}_mean'] = valid_x[c].map(target_mean)
        enc_cols.append(f'enc_{c}_mean')
        del tmp_df, target_mean
        gc.collect()

    return train_x, valid_x, enc_cols
