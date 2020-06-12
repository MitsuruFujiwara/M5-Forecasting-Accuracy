
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

    # TODO: is_zero mean & days after last sales

    return df

# target encoding
def target_encoding(train_df,test_df):
    # cols to encode
    cols_id = ['item_id','cat_id','dept_id','store_id','state_id']
    enc_cols = []
    for c in cols_id:
        df_grouped = train_df[[c,'demand']].groupby(c)

        train_df[f'enc_{c}_mean'] = train_df[c].map(df_grouped.mean()['demand'])
        test_df[f'enc_{c}_mean'] = test_df[c].map(df_grouped.mean()['demand'])
        enc_cols.append(f'enc_{c}_mean')

    return train_df, test_df, enc_cols

# target encoding for cv
def target_encoding_cv(train_df,train_idx,valid_idx):
    cols_id = ['item_id','cat_id','dept_id','store_id','state_id']
    enc_cols = []
    print('target encoding...')
    for c in tqdm(cols_id):
        df_grouped = train_df[[c,'demand']].iloc[train_idx].groupby(c)
        train_df[f'enc_{c}_mean'] = 0
        train_df.iloc[train_idx][f'enc_{c}_mean'] += train_df.iloc[train_idx][c].map(df_grouped.mean()['demand'])
        train_df.iloc[valid_idx][f'enc_{c}_mean'] += train_df.iloc[valid_idx][c].map(df_grouped.mean()['demand'])
        enc_cols.append(f'enc_{c}_mean')

    return train_df, enc_cols
