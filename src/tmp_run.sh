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
python 106_aggregation_foods.py
python 107_aggregation_household.py
python 108_aggregation_hobbies.py
