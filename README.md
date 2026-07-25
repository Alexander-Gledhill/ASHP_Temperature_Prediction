This repository contains code for both temperature prediction models used in the paper: "Towards Demand-Response Control of Residential Heat Pumps: A Physics-Informed Probabilistic Framework for Indoor Temperature Forecasting." 

3R2C_model.py -> Resistance-capacitance model used to generate "baseline forecast."
hybrid_model.py -> Hybrid formulation training an LSTM on the residual predictive error.
