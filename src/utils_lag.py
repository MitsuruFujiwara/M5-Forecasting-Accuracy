
import gc

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

    return df

# TODO: # helper for making lag features with ewm
def make_lags_ewm(df,days=28):
    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm([0,7,14,21,28]):
        df[f'demand_lag_{days}_{i}'] = df_grouped.shift(days+i)

    print('Add rolling aggs...')
    for i, alpha in tqdm(zip([7,14,21,28],[0.1,0.075,0.05,0.025])):
        df[f'demand_rolling_mean_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).ewm(alpha=alpha))
        df[f'demand_rolling_std_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).std())
        df[f'demand_rolling_max_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).max())
        df[f'demand_rolling_min_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).min())
        df[f'demand_rolling_skew_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).skew())
        df[f'demand_rolling_kurt_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).kurt())

    del df_grouped
    gc.collect()

    return df

# TODO: # target encoding
def target_encoding(train_x,valid_x,train_y):
    # cols to encode
    cols_id = ['item_id','cat_id','dept_id']
    enc_cols = []
    for c in cols_id:
        df_grouped = train_df[[c,'demand']].groupby(c)['demand']

        train_df[f'enc_{c}_mean'] = train_df[c].map(df_grouped)
        test_df[f'enc_{c}_mean'] = test_df[c].map(df_grouped)
        enc_cols.append(f'enc_{c}_mean')

    return train_df, test_df, enc_cols
