#!/bin/sh
#python 102_aggregation_21days.py
python 103_aggregation_14days.py
python 104_aggregation_7days.py

python 202_cv_lgbm_21days.py
python 203_cv_lgbm_14days.py
python 204_cv_lgbm_7days.py

python 302_train_lgbm_21days.py
python 303_train_lgbm_14days.py
python 304_train_lgbm_7days.py

python 206_cv_lgbm_group_k_fold.py
python 210_cv_lgbm_group_k_fold_foods.py
python 211_cv_lgbm_group_k_fold_household.py
python 212_cv_lgbm_group_k_fold_hobbies.py
