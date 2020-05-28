#!/bin/sh
python 201_cv_lgbm_28days.py
python 202_cv_lgbm_21days.py
python 203_cv_lgbm_14days.py
python 204_cv_lgbm_7days.py

python 301_train_lgbm_28days.py
python 302_train_lgbm_21days.py
python 303_train_lgbm_14days.py
python 304_train_lgbm_7days.py

python 401_predict.py
