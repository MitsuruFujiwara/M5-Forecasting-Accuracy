#!/bin/sh
cd ../feats
rm *.feather
cd ../src
python 101_aggregation_28days.py
python 102_aggregation_21days.py
python 103_aggregation_14days.py
python 104_aggregation_7days.py
python 106_aggregation_foods.py
python 107_aggregation_household.py
python 108_aggregation_hobbies.py
