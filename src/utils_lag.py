
import gc

from tqdm import tqdm

#==============================================================================
# utils for lag features
#==============================================================================

DAYS_PRED = 1

# helper for making lag features
def make_lags(df):
    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm([0,7,14,28]):
        df[f'demand_lag_{i}'] = df_grouped.shift(DAYS_PRED+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,28]):
        df[f'demand_rolling_mean_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).mean())
        df[f'demand_rolling_std_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).std())
        df[f'demand_rolling_max_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).max())
        df[f'demand_rolling_min_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).min())
        df[f'demand_rolling_skew_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).skew())
        df[f'demand_rolling_kurt_{i}'] = df_grouped.transform(lambda x: x.shift(DAYS_PRED).rolling(i).kurt())

    del df_grouped
    gc.collect()

    return df
