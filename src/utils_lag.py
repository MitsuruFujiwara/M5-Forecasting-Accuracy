
import gc

from tqdm import tqdm

#==============================================================================
# utils for lag features
#==============================================================================

# 28days lag
def make_lags(df,days=28):
    # lag features
    df_grouped = df[['id','demand']].groupby(['id'])['demand']

    print('Add lag features...')
    for i in tqdm([0,1,2,3,4,5,6,7,14,21,28]):
        df[f'demand_lag_{days}_{i}'] = df_grouped.shift(days+i)

    print('Add rolling aggs...')
    for i in tqdm([7,14,21,28,56]):
        df[f'demand_rolling_mean_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).mean())
        df[f'demand_rolling_std_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).std())
        df[f'demand_rolling_max_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).max())
        df[f'demand_rolling_min_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).min())
        df[f'demand_rolling_skew_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).skew())
        df[f'demand_rolling_kurt_{days}_{i}'] = df_grouped.transform(lambda x: x.shift(days).rolling(i).kurt())

    del df_grouped
    gc.collect()

    return df

# TODO:target encoding
def target_encoding(train_df,test_df):
    return
