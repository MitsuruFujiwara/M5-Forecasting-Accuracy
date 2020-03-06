
import feather
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from utils import save2pkl, line_notify

#===============================================================================
# preprocess sell calendar
#===============================================================================

warnings.simplefilter(action='ignore')

def main(is_eval=False):
    # load csv
    df = pd.read_csv('../input/calendar.csv')

    # TODO: feature engineering

    # save pkl
    save2pkl('../feats/calendar.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
