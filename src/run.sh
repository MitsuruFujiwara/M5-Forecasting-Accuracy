#!/bin/sh
#cd ../feats
#rm *.feather
#rm *.pkl
#rm sales/*.pkl
#rm sales_diff/*.pkl
#cd ../src

#python 001_sales.py
#python 002_calendar.py
#python 003_sell_prices.py

#python 101_aggregation_28days.py
#python 102_aggregation_21days.py
#python 103_aggregation_14days.py
#python 104_aggregation_7days.py
#python 105_aggregation_diff.py
#python 106_aggregation_foods.py
#python 107_aggregation_household.py
#python 108_aggregation_hobbies.py

#python 201_cv_lgbm_28days.py
#python 202_cv_lgbm_21days.py
#python 203_cv_lgbm_14days.py
#python 204_cv_lgbm_7days.py
#python 205_cv_lgbm_diff.py
#python 206_cv_lgbm_group_k_fold.py
#python 207_cv_lgbm_foods.py
python 208_cv_lgbm_households.py
python 209_cv_lgbm_hobbies.py

#python 301_train_lgbm_28days.py
#python 302_train_lgbm_21days.py
#python 303_train_lgbm_14days.py
#python 304_train_lgbm_7days.py
#python 305_train_lgbm_diff.py
python 306_train_lgbm_foods.py
python 307_train_lgbm_household.py
python 308_train_lgbm_hobbies.py

#python 401_predict.py
python 404_predict_cat_id.py
