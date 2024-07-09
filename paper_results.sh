#!/bin/bash

# Single fit Dummy  
python example_fit.py --dummy=1 --name=DUMMY
# Single fit LR unfair
python example_fit.py --lr=1 --unfair=1 --name=LR_UNFAIR
# Single fit LR unaware
python example_fit.py --lr=1 --name=LR_UNAWARE
# Single LR with fairness penalty
python example_fit.py --lr_fair=1 --lambda_fair=3.0 --name=LR_FAIR

# Single XGBoost fit for features utility and relative plots
python example_fit.py --xgb=1 --name=XGB_UNAWARE
python example_fit.py --xgb=1 --unfair=1 --name=XGB_UNFAIR

# Generate LR results with different level of fairness penalty
python example_simulations_LR.py --lr_fair=1 --lambda_fair=0.1 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=0.316 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=0.5 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=1.0 --name=LR_FAIR
python example_simulations_LR.py --lr_fair=1 --lambda_fair=10.0 --name=LR_FAIR
# Generate LR results without fairness penalty
python example_simulations_LR.py --lr=1 --name=LR

# For post-processing of the results see the notebook: dataset_analysis.ipynb
