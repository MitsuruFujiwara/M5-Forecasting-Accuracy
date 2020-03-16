
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from utils import save2pkl, line_notify

#===============================================================================
# preprocess sell prices
#===============================================================================

warnings.simplefilter(action='ignore')

def main(is_eval=False):
    # load csv
    df = pd.read_csv('../input/sell_prices.csv')

    # TODO: feature engineering
    # TODO: moving average?

    # save pkl
    save2pkl('../feats/sell_prices.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
