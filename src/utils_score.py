
import pandas as pd
import numpy as np

from typing import Union
from tqdm import tqdm

#==============================================================================
# utils for scores
#==============================================================================

VERBOSE=False

# function for evaluating WRMSSEE
# ref: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834
class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(self.group_ids):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def get_scale(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        return getattr(self, f'lv{lv}_scale')

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())
        if VERBOSE:
            print(np.round(all_scores,3))
        return np.mean(all_scores)

    def full_score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())
        print(np.round(all_scores,3))
        return np.mean(all_scores)

# WRMSSE for LightGBM
# https://www.kaggle.com/kyakovlev/m5-custom-validation
class WRMSSEForLightGBM(WRMSSEEvaluator):

    def feval(self, preds, dtrain):
        preds = preds.reshape(self.valid_df[self.valid_target_columns].shape, order='F') #.transpose()
        score = self.score(preds)
        return 'WRMSSE', score, False

    def full_feval(self, preds, dtrain):
        preds = preds.reshape(self.valid_df[self.valid_target_columns].shape, order='F') #.transpose()
        score = self.full_score(preds)
        return 'WRMSSE', score, False

# get evaluator for cross validation
def get_evaluators():
    # load csv files
    df = pd.read_csv('../input/sales_train_evaluation.csv')
    calendar = pd.read_csv('../input/calendar.csv')
    prices = pd.read_csv('../input/sell_prices.csv')

    lgb_evaluators = []

    # fold1
    df_train1 = df.iloc[:, :-28]
    df_valid1 = df.iloc[:, -28:]
    evaluator1 = WRMSSEForLightGBM(df_train1, df_valid1, calendar, prices)
    lgb_evaluators.append(evaluator1)
    del df_train1, df_valid1

    # fold2
    df_train2 = df.iloc[:, :-28*2]
    df_valid2 = df.iloc[:, -28*2:-28]
    evaluator2 = WRMSSEForLightGBM(df_train2, df_valid2, calendar, prices)
    lgb_evaluators.append(evaluator2)
    del df_train2, df_valid2

    # fold3
    df_train3 = df.iloc[:, :-365]
    df_valid3 = df.iloc[:, -365:-365+28]
    evaluator3 = WRMSSEForLightGBM(df_train3, df_valid3, calendar, prices)
    lgb_evaluators.append(evaluator3)
    del df_train3, df_valid3

    return lgb_evaluators

# calc cv score
def calc_score_cv(oof,end_train=1941):
    # load csv files
    df = pd.read_csv('../input/sales_train_evaluation.csv')
    calendar = pd.read_csv('../input/calendar.csv')
    prices = pd.read_csv('../input/sell_prices.csv')
    sample_sub = pd.read_csv('../input/sample_submission.csv')

    # add order
    sample_sub['order'] = range(sample_sub.shape[0])

    # split oof by folds
    cols_valid1 = [f'd_{c}' for c in range(end_train-28+1,end_train+1)]
    cols_valid2 = [f'd_{c}' for c in range(end_train-28*2+1,end_train-28+1)]
    cols_valid3 = [f'd_{c}' for c in range(end_train-365+1,end_train-365+28+1)]

    # sort
    oof = oof.merge(sample_sub[['id','order']],on='id')
    oof.sort_values('order',inplace=True)
    oof.drop(['id','order'],axis=1,inplace=True)
    oof.reset_index(drop=True,inplace=True)

    # calc scores
    df_train1 = df.iloc[:, :-28]
    df_valid1 = df.iloc[:, -28:]
    evaluator1 = WRMSSEEvaluator(df_train1, df_valid1, calendar, prices)
    score1 = evaluator1.score(oof[cols_valid1])

    df_train2 = df.iloc[:, :-28*2]
    df_valid2 = df.iloc[:, -28*2:-28]
    evaluator2 = WRMSSEEvaluator(df_train2, df_valid2, calendar, prices)
    score2 = evaluator2.score(oof[cols_valid2])

    df_train3 = df.iloc[:, :-365]
    df_valid3 = df.iloc[:, -365:-365+28]
    evaluator3 = WRMSSEEvaluator(df_train3, df_valid3, calendar, prices)
    score3 = evaluator3.score(oof[cols_valid3])

    # get scores
    scores = [score1,score2,score3]

    return scores
