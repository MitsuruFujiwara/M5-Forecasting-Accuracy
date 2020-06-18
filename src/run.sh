#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
rm sales/*.pkl
cd ../src

python 001_sales.py
python 002_calendar.py
python 003_sell_prices.py

python 101_aggregation_28days.py
python 102_aggregation_21days.py
python 103_aggregation_14days.py
python 104_aggregation_7days.py
#python 105_aggregation_diff.py
python 106_aggregation_foods.py
python 107_aggregation_household.py
python 108_aggregation_hobbies.py
python 109_aggregation_weekday.py
python 110_aggregation_holiday.py

python 201_cv_lgbm_28days.py
python 202_cv_lgbm_21days.py
python 203_cv_lgbm_14days.py
python 204_cv_lgbm_7days.py
#python 205_cv_lgbm_diff.py
#python 206_cv_lgbm_group_k_fold.py
python 207_cv_lgbm_foods.py
python 208_cv_lgbm_household.py
python 209_cv_lgbm_hobbies.py
python 210_cv_lgbm_group_k_fold_foods.py
python 211_cv_lgbm_group_k_fold_household.py
python 212_cv_lgbm_group_k_fold_hobbies.py
python 213_cv_lgbm_group_k_fold_28days.py
python 214_cv_lgbm_group_k_fold_21days.py
python 215_cv_lgbm_group_k_fold_14days.py
python 216_cv_lgbm_group_k_fold_7days.py
python 217_cv_lgbm_weekday.py
python 218_cv_lgbm_holiday.py
python 219_cv_lgbm_group_k_fold_weekday.py
python 220_cv_lgbm_group_k_fold_holiday.py

python 301_train_lgbm_28days.py
python 302_train_lgbm_21days.py
python 303_train_lgbm_14days.py
python 304_train_lgbm_7days.py
#python 305_train_lgbm_diff.py
python 306_train_lgbm_foods.py
python 307_train_lgbm_household.py
python 308_train_lgbm_hobbies.py
python 309_train_lgbm_weekday.py
python 310_train_lgbm_holiday.py

python 401_predict_28days.py
python 404_predict_cat_id.py
python 405_predict_cat_id_group_k_fold.py
python 407_predict_weekly.py
python 408_predict_group_k_fold.py
python 409_predict_weekly_group_k_fold.py
python 410_predict_holiday.py
python 411_predict_holiday_group_k_fold.py

python 501_blend.py
