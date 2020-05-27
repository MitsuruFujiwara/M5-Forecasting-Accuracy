#!/bin/sh
python 301_train_lgbm_28days.py
python 302_train_lgbm_21days.py
python 303_train_lgbm_14days.py
python 304_train_lgbm_7days.py

python 401_predict.py
