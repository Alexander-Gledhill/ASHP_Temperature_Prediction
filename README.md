# ASHP_Temperature_Prediction
Indoor temperature prediction model used for air-source heat pump (ASHP) data presented in the paper "Indoor Temperature Prediction for Residential Heat Pumps: A Physics-Informed Machine Learning Approach"

This model combines:
1.) A 3R2C resistance-capacitance (RC) thermal network to capture building physics
2.) Automatic Differentiation Variational Inference (ADVI) to calibrate unknown parameters for each home
3.) A hybrid Bayesian LSTM extension, which predicts the residual (difference between RC prediction and true indoor temperature)
