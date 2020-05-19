#!/bin/sh
#cd ../feats
#rm *.feather
#rm *.pkl
#rm sales/*.pkl
#cd ../src
#python 001_sales.py
#python 002_calendar.py
#python 003_sell_prices.py
#python 101_aggregation.py
#python 201_train_lgbm_cv.py
python 202_train_lgbm_all_data.py
python 301_predict_recursive.py
