#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
rm sales/*.pkl
cd ../src

python 001_sales.py
python 002_calendar.py
python 003_sell_prices.py

python 109_aggregation_weekday.py
python 110_aggregation_holiday.py

python 219_cv_lgbm_group_k_fold_weekday.py
python 220_cv_lgbm_group_k_fold_holiday.py

python 411_predict_holiday_group_k_fold.py
