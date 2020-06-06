
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
# weekly prediction
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load submission files
    sub1 = pd.read_csv("../output/submission_cat_id.csv",index_col=0) # 0.54652
    sub2 = pd.read_csv("../output/submission_lgbm_group_k_fold.csv",index_col=0) # 0.54174

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
