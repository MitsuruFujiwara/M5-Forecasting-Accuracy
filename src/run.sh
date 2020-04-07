#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
cd ../src
python 001_sales.py > ../log/001_sales.py.txt
python 002_calendar.py > ../log/002_calendar.py.txt
python 003_sell_prices.py > ../log/003_sell_prices.py.txt
python 101_aggregation.py > ../log/101_aggregation.txt
python 202_train_lgbm_holdout.py > ../log/202_train_lgbm_holdout.py.txt
