# Hybrid RC–LSTM Model for Indoor Temperature Prediction
Requires: Python 3.9+, PyTorch, NumPy, Pandas, Matplotlib

This repository contains code for a **hybrid physics–machine learning model** to forecast indoor temperature in buildings.  
Model results were featured in the paper "Indoor Temperature Prediction for Residential Heat Pumps: A Physics-Informed Machine Learning Approach"

The model combines:
1. A **3R2C resistance–capacitance (RC) thermal network** to capture building physics.
2. **Automatic Differentiation Variational Inference (ADVI)** for Bayesian calibration of unknown RC parameters.
3. A **Residual LSTM with Bayesian output layer** to learn and quantify uncertainty in residual dynamics not explained by the RC model.

---

## 1. Background

### 1.1 The 3R2C Model

The building thermal zone is approximated as a **3R2C network** consisting of:
- Indoor air capacitance \( C_{in} \),
- Envelope capacitance \( C_{en} \),
- Resistances between nodes:
  - \( R_{ia} \): indoor to ambient,
  - \( R_{ie} \): indoor to envelope,
  - \( R_{ea} \): envelope to ambient.

The thermal balance equations are:

The thermal balance equations are:

$$
C_{in}\,\frac{dT_{in}}{dt}
=
\frac{T_e - T_{in}}{R_{ie}}
+ \frac{T_a - T_{in}}{R_{ia}}
+ a_{\text{sol,in}}\, Q_{\text{sol}}
+ a_{\text{int,in}}\, Q_{\text{int}}
+ Q_{ah}
$$

$$
C_{en}\,\frac{dT_e}{dt}
=
\frac{T_{in} - T_e}{R_{ie}}
+ \frac{T_a - T_e}{R_{ea}}
+ a_{\text{sol,en}}\, Q_{\text{sol}}
+ a_{\text{int,en}}\, Q_{\text{int}}
$$

where:
- \(T_{in}\): indoor air temperature,
- \(T_e\): envelope temperature,
- \(T_a\): ambient (outdoor) temperature,
- \(Q_{sol}\): solar gains (irradiance × glazing area × transmittance),
- \(Q_{int}\): internal gains,
- \(Q_{ah}\): HVAC/air handling unit heat flux,
- \(a_{sol,in}, a_{sol,en}, a_{int,in}, a_{int,en}\): absorption distribution coefficients (e.g. how much of the total solar gain is shared between indoor and envelope nodes).

---

### 1.2 Discretisation: Backward Euler

The state vector is:

\[
x = 
\begin{bmatrix}
T_e \\
T_{in}
\end{bmatrix},
\quad
u =
\begin{bmatrix}
T_a \\
Q_{sol} \\
Q_{int} \\
Q_{ah}
\end{bmatrix}
\]

The continuous dynamics can be written as:

\[
\dot{x}(t) = A x(t) + B u(t)
\]

Backward Euler integration gives a stable discrete-time update:

\[
(I - \Delta t \, A) \, x_{k+1} = x_k + \Delta t \, (B u_{k+1})
\]

where:
- \(\Delta t\) is the timestep (30 minutes in our dataset),
- \(A, B\) are system matrices derived from resistances and capacitances.

This formulation allows a computationally efficient simulation in PyTorch.

---

## 2. Bayesian Calibration with ADVI

Initial parameter estimates are treated as **priors**.  
These were calculated as analytical approximations for each home.
In unconstrained space (e.g. the set of all real numbers):
- Positive-only parameters (\(R, C, \sigma\)) use an exponential transform:  
  \[
  z \sim \mathcal{N}(\mu, \sigma^2), \quad R = \exp(z)
  \]
- Fractional parameters (\(a \in (0,1)\)) use the sigmoid transform:  
  \[
  a = \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

ADVI fits the posterior by maximising the **Evidence Lower Bound (ELBO):**

\[
\text{ELBO} = \mathbb{E}_q[\log p(D \mid \theta)] - \text{KL}(q(\theta)\,\|\,p(\theta))
\]

where:
- \(q(\theta)\) is the variational Gaussian approximation of the posterior,
- \(p(\theta)\) is the prior.

This yields calibrated RC parameters with uncertainty estimates.

---

## 3. Residual Learning with Bayesian LSTM

The RC model alone cannot capture all dynamics (e.g. sensor noise, measurment error, more complex dynamic phenomena).  
We define the **residual heating/cooling rate** as:

\[
y_t = \frac{\Delta T_{in,t}}{\Delta t} - \frac{\Delta T_{in,t}^{RC}}{\Delta t}
\]

where:
- \(\Delta T_{in,t}\): observed indoor temperature change,
- \(\Delta T_{in,t}^{RC}\): RC-predicted change.

### LSTM Residual Model

An LSTM receives a **48-step input sequence (24 hours)** of features:
- Weather: \(T_a\), irradiance, solar angles,
- HVAC/internal gains,
- RC predictions,
- Cyclical encodings of hour-of-day and day-of-week.

It outputs a predictive distribution over \(y_t\).

### Bayesian Output Layer

A **Bayesian linear layer** provides uncertainty-aware predictions:

\[
\hat{y}_t \sim \mathcal{N}(\mu_t, \sigma_t^2)
\]

Weights are treated as random variables with Gaussian priors.  
The loss combines:
- Gaussian negative log-likelihood (NLL),
- KL-divergence between approximate posterior and prior.

This balances **data fit** and **regularisation**.

---

## 4. Monte Carlo Prediction

At test time, uncertainty is quantified via **Monte Carlo sampling**:
1. Draw \(M\) stochastic forward passes through the Bayesian LSTM.
2. Integrate predicted residuals into the RC baseline:

\[
T_{in,t}^{\text{pred}} = T_{in,t}^{RC} + r_t
\]

3. Collect ensemble statistics (mean, standard deviation) to form **predictive bands**.

---

## 5. Metrics

Performance is evaluated using:
- **RMSE** (Root Mean Squared Error):
  \[
  \text{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^N (T_{in,t}^{\text{pred}} - T_{in,t}^{\text{true}})^2}
  \]
- **CVRMSE** (Coefficient of Variation of RMSE):
  \[
  \text{CVRMSE} = 100 \times \frac{\text{RMSE}}{\bar{T}_{in}}
  \]
---
## 6. Preprocessed Data

While not provided here (for the sake of data protection) this model requires data for the:
- Thermal output power of the ASHP [W]
- Total direct irradiance [W/m^2], window area [m^2] and transmittance value (for solar gain calculation)
- Internal gain [W] (we used the expected internal gain based on occupancy schedules listed by ISO Standards)
- Fabric resistance [K/W] (e.g. the reciprocal of the area-averaged mean thermal transmittance for the envelope)
- Ventilation resistance [K/W] (calculating using Home Energy Model methodology, dynamic value)
- Solar elevation/azimuth angles at each timestep (LSTM only)
- These must be aggregated into a consistent time interval, with timesteps listed in the index column (here we chose 30 min intervals)
---

## 7. Repository Structure

- `3R2C.py`: Physics-only RC model + ADVI calibration.
- `RC-LSTM.py`: Hybrid RC–LSTM implementation.
- `README.md`: This document.

---

## 8. References

- State-space building thermal models: Clarke, J. (2001). *Energy Simulation in Building Design*.  
- ADVI: Kucukelbir et al. (2017). *Automatic Differentiation Variational Inference*.  
- Bayesian deep learning: Blundell et al. (2015). *Weight Uncertainty in Neural Networks*.  

---

## 9. Usage

```bash
# Train RC model parameters with ADVI
python 3R2C.py

# Train hybrid RC–LSTM model
python hybrid_rc_lstm.py
