#!/bin/bash

cd cfexplainer

# =========================
# TRAIN
# =========================
python main.py --do_train --cuda_id 0

# =========================
# TEST
# =========================
python main.py --do_test --cuda_id 0

# =========================
# GENERATE EXPLANATIONS
# =========================
python main.py --do_explain --KM 8 --cuda_id 0