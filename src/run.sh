#!/bin/sh
cd ../feats
rm *.feather
rm *.pkl
cd ../src
python 001_sales.py
python 002_sell_prices.py
python 003_calendar.py
python 101_aggregation.py
