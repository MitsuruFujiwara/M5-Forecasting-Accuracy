#!/bin/sh
python 404_predict_cat_id.py
python 405_predict_cat_id_group_k_fold.py
python 501_blend.py
