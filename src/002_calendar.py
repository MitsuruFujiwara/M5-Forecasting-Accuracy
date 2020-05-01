
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

    # to datetime
    df['date'] = pd.to_datetime(df['date'])

    # factorize numerical columns
    cols_string = ['event_name_1','event_type_1','event_name_2','event_type_2']
    for c in cols_string:
        df[c], _ = pd.factorize(df[c])
        df[c].replace(-1,np.nan,inplace=True)

    # seasonality
    df['seasonality'] = np.cos(np.pi*(df['date'].dt.dayofyear/366*2-1))

    # drop string columns
    df.drop('weekday',axis=1,inplace=True)

    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.weekofyear

    # TODO: feature engineering

    # save pkl
    save2pkl('../feats/calendar.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
