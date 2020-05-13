#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
rm sales/*.pkl
cd ../src
python 001_sales.py
python 002_calendar.py
python 003_sell_prices.py
python 101_aggregation.py
python 204_train_lgbm_custom_cv.py
python 203_train_lgbm_all_data.py
