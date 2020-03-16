
import feather
import gc
import json
import numpy as np
import pandas as pd
import sys
import warnings

from utils import loadpkl, to_feature, line_notify
from utils import removeCorrelatedVariables, removeMissingVariables, reduce_mem_usage

#===============================================================================
# aggregation
#===============================================================================

warnings.simplefilter(action='ignore')

def main():
    # load pkls
    df = loadpkl('../feats/sales.pkl')
    df_calendar = loadpkl('../feats/calendar.pkl')
    df_sell_prices = loadpkl('../feats/sell_prices.pkl')

    # merge
    df = df.merge(df_calendar, on='d',how='left')
    df = df.merge(df_sell_prices, on=['store_id','item_id','wm_yr_wk'],how='left')

    del df_calendar, df_sell_prices
    gc.collect()

    # use data from 2015-1
    df = df[df['date'] >= '2015-1-1']

    # reduce memory usage
    df = reduce_mem_usage(df)

    # save as feather
    to_feature(df, '../feats')

    # save feature name list
    features_json = {}
    features_json['features'] = df.columns.tolist()
    with open('../configs/all_features.json', 'w') as f:
        json.dump(features_json, f, indent=4)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
