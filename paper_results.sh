#!/bin/bash

# Generate latex tables and plots for data summaries
python dataset_summary.py

# Single XGBoost fit for features utility and relative plots
python example_fit.py --xgb=1 --name=XGB_UNAWARE
python example_fit.py --xgb=1 --unfair=1 --name=XGB_UNFAIR
python plots_xgb.py

# Generate LR results with different level of fairness penalty
python example_simulations_LR.py --lr_fair=1 --lambda_fair=0.1 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=0.316 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=0.5 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=1.0 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=10.0 --name=LR_FAIR

# Generate LR results without fairness penalty
python example_simulations_LR.py --lr=1 --name=LR

# Generate LR (and fair version) related plots
python plots_lr.py



