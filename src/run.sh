#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
rm sales/*.pkl
rm sales_diff/*.pkl
cd ../src

python 001_sales.py
python 002_calendar.py
python 003_sell_prices.py

python 101_aggregation_28days.py
python 102_aggregation_21days.py
python 103_aggregation_14days.py
python 104_aggregation_7days.py

python 201_cv_lgbm_28days.py
python 202_cv_lgbm_21days.py
python 203_cv_lgbm_14days.py
python 204_cv_lgbm_7days.py

python 301_train_lgbm_28days.py
python 302_train_lgbm_21days.py
python 303_train_lgbm_14days.py
python 304_train_lgbm_7days.py

python 401_predict.py
