
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from utils import save2pkl, line_notify

#===============================================================================
# preprocess sales
#===============================================================================

warnings.simplefilter(action='ignore')

def main(is_eval=False):
    # load csv
    df = pd.read_csv('../input/sales_train_validation.csv')

    # merge test data
    if is_eval:
        df_test = pd.read_csv('../input/sales_train_evaluation.csv')
        df = df.append(df_test)

        del df_test
        gc.collect()

    # TODO: feature engineering

    # save pkl
    save2pkl('../feats/sales.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))


if __name__ == '__main__':
    main(is_eval=False)
