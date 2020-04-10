#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
cd ../src
echo python 001_sales.py > ../log/001_sales.py.txt
echo python 002_calendar.py > ../log/002_calendar.py.txt
echo python 003_sell_prices.py > ../log/003_sell_prices.py.txt
echo python 101_aggregation.py > ../log/101_aggregation.txt
echo python 202_train_lgbm_holdout.py > ../log/202_train_lgbm_holdout.py.txt
