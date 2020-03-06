
import feather
import gc
import json
import numpy as np
import pandas as pd
import sys
import warnings

from utils import loadpkl, to_feature, removeCorrelatedVariables, removeMissingVariables

#===============================================================================
# aggregation
#===============================================================================

warnings.simplefilter(action='ignore')

def main():
    # load pkls
    df = loadpkl('../feats/sales.pkl')
    df_sell_prices = loadpkl('../feats/sell_prices.pkl')
    df_calendar = loadpkl('../feats/calendar.pkl')

    # TODO: merge & feature engineering

    # save as feather
    to_feature(df, '../features')

    # save feature name list
    features_json = {}
    features_json['features'] = df.columns.tolist()
    with open('../feats/all_features.json', 'w') as f:
        json.dump(features_json, f, indent=4)

if __name__ == '__main__':
    main()
