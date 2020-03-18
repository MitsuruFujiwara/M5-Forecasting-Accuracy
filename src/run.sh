#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
cd ../src
python 001_sales.py
python 002_calendar.py
python 003_sell_prices.py
python 101_aggregation.py
#python 201_train_lgbm.py
