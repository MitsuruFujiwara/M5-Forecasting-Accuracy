
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
# blending
#==============================================================================

warnings.filterwarnings('ignore')

def netflix(es, ps, e0, la=.0001):
    """Combine predictions with the optimal weights to minimize RMSE.
    Args:
        es (list of float): RMSEs of predictions
        ps (list of np.array): predictions
        e0 (float): RMSE of all zero prediction
        la (float): lambda as in the ridge regression
    Returns:
        (tuple):
            - (np.array): ensemble predictions
            - (np.array): weights for input predictions
    """
    m = len(es)
    n = len(ps[0])

    X = np.stack(ps).T
    pTy = .5 * (n * e0**2 + (X**2).sum(axis=0) - n * np.array(es)**2)

    w = np.linalg.pinv(X.T.dot(X) + la * n * np.eye(m)).dot(pTy)
    return X.dot(w), w

def main():
    # load submission files
    sub1 = pd.read_csv("../output/submission_cat_id.csv",index_col=0) # 0.54652
    sub2 = pd.read_csv("../output/submission_lgbm_group_k_fold.csv",index_col=0) # 0.54174

    # TODO: calc weights by ridge regression
    """
    es = [
        0.303460,
        0.329844,
        0.303750
    ]
    ps = [
        lgbm_0,
        lgbm_1,
        cat_0
    ]
    e0 = 0.926908

    pred, w = netflix(es, ps, e0, la=.0001)
    print(w)
    """

    # averaging
    sub = 0.5*sub1 + 0.5*sub2

    # postprocesssing
    cols_f = [f'F{i}' for i in range(1,29)]
    sub.loc[:,cols_f] = sub[cols_f].where(sub[cols_f]>0,0)

    # reset index
    sub.reset_index(inplace=True)

    # save csv
    sub.to_csv(submission_file_name, index=False)

    # submission by API
    submit(submission_file_name, comment='model405 blending')

if __name__ == '__main__':
    submission_file_name = "../output/submission_blend.csv"
    main()
