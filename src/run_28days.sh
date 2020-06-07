#!/bin/sh
cd ../feats
rm f101_*.feather
rm *.pkl
rm sales/*.pkl
cd ../src

python 001_sales.py
python 002_calendar.py
python 003_sell_prices.py

python 101_aggregation_28days.py

python 201_cv_lgbm_28days.py
python 301_train_lgbm_28days.py
python 401_predict_28days.py
