#!/bin/bash
python3 00_fetch_abc_data.py
python3 01_clean_and_filter.py
python3 02_generate_features.py
python3 03_train_test_split.py
python3 04_model_train_eval.py
python3 05_feature_importance.py
