
import gc

from tqdm import tqdm

#==============================================================================
# utils for lag features
#==============================================================================

DAYS_PRED = 28

# make lags for weekly prediction
def make_lags_weekly(df):
    # grouped df
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm([7,14,21,28]):
        df[f'demand_lag_{i}'] = df_grouped.shift(i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,21,28]):
        df_grouped_lag = df[['id',f'demand_lag_{i}']].groupby(['id'])[f'demand_lag_{i}']
        for w in [7,14,21,28]:
            df[f'demand_rolling_mean_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).mean())
            df[f'demand_rolling_std_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).std())
            df[f'demand_rolling_max_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).max())
            df[f'demand_rolling_min_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).min())
            df[f'demand_rolling_skew_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).skew())
            df[f'demand_rolling_kurt_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).kurt())

# 28days lag
def make_lags_28(df):
    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm([0,7,14,21,28]):
        df[f'demand_lag_{i}'] = df_grouped.shift(DAYS_PRED+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,21,28]):
        df[f'demand_rolling_mean_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).mean())
        df[f'demand_rolling_std_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).std())
        df[f'demand_rolling_max_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).max())
        df[f'demand_rolling_min_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).min())
        df[f'demand_rolling_skew_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).skew())
        df[f'demand_rolling_kurt_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).kurt())

    del df_grouped
    gc.collect()

    return df

# helper for making lag features
def make_lags(df):
    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm([7,28]):
        df[f'demand_lag_{i}'] = df_grouped.shift(i)

        # add rolling features
        df_grouped_lag = df[['id',f'demand_lag_{i}']].groupby(['id'])[f'demand_lag_{i}']
        for w in [7,28]:
            df[f'demand_rolling_mean_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).mean())
            df[f'demand_rolling_std_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).std())
            df[f'demand_rolling_max_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).max())
            df[f'demand_rolling_min_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).min())
            df[f'demand_rolling_skew_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).skew())
            df[f'demand_rolling_kurt_{i}_{w}'] = df_grouped_lag.transform(lambda x: x.rolling(w).kurt())

        del df_grouped_lag
        gc.collect()

    del df_grouped
    gc.collect()

    return df

# TODO:target encoding
def target_encoding(train_df,test_df):
    return
