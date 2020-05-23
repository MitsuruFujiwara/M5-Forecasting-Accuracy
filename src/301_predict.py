
import gc
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings

from glob import glob
from tqdm import tqdm

from utils import submit
from utils import FEATS_EXCLUDED, COLS_TEST1, COLS_TEST2
from utils_lag import make_lags

#==============================================================================
# Recursive prediction
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load feathers
    files = sorted(glob('../feats/*.feather'))
    df = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)], axis=1)
    df = df[configs['features']]
    feats = [f for f in df.columns if f not in FEATS_EXCLUDED]

    # load model
    reg = lgb.Booster(model_file='../output/lgbm_all_data.txt')

    if not debug:
        # save out of fold prediction
        train_df.loc[:,'demand'] = oof_preds
        train_df = train_df.reset_index()
        train_df[['id', 'demand']].to_csv(oof_file_name, index=False)

        # reshape prediction for submit
        test_df.loc[:,'demand'] = sub_preds
        test_df = test_df.reset_index()
        preds = test_df[['id','d','demand']].reset_index()
        preds = preds.pivot(index='id', columns='d', values='demand').reset_index()

        # split test1 / test2
        preds1 = preds[['id']+COLS_TEST1]
        preds2 = preds[['id']+COLS_TEST2]

        # change column names
        preds1.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
        preds2.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]

        # replace test2 id
        preds2['id']= preds2['id'].str.replace('_validation','_evaluation')

        # merge
        preds = preds1.append(preds2)

        # save csv
        preds.to_csv(submission_file_name, index=False)

        # submission by API
        submit(submission_file_name, comment='model203 cv: %.6f' % full_rmse)

    # reshape prediction for submit
    preds = test_df[['id','d','demand']].reset_index()
    preds = preds.pivot(index='id', columns='d', values='demand').reset_index()

    # split test1 / test2
    preds1 = preds[['id']+COLS_TEST1]
    preds2 = preds[['id']+COLS_TEST2]

    # change column names
    preds1.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
    preds2.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]

    # replace test2 id
    preds2['id']= preds2['id'].str.replace('_validation','_evaluation')

    # merge
    preds = preds1.append(preds2)

    # save csv
    preds.to_csv(submission_file_name, index=False)

    # submission by API
    submit(submission_file_name, comment='model301 recursive prediction')

if __name__ == '__main__':
    submission_file_name = "../output/submission_lgbm.csv"
    configs = json.load(open('../configs/202_lgbm_all_data.json'))
    main()
