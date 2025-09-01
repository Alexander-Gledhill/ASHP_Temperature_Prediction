# Hybrid RC–LSTM Model for Indoor Temperature Prediction
Requires: Python 3.9+, PyTorch, NumPy, Pandas, Matplotlib

This repository contains code for a **hybrid physics–machine learning model** to forecast indoor temperature in buildings.  
Model results were featured in the paper *Indoor Temperature Prediction for Residential Heat Pumps: A Physics-Informed Machine Learning Approach*.

The model combines:
1. A **3R2C resistance–capacitance (RC) thermal network** to capture building physics.
2. **Automatic Differentiation Variational Inference (ADVI)** for Bayesian calibration of unknown RC parameters.
3. A **Residual LSTM with Bayesian output layer** to learn and quantify uncertainty in residual dynamics not explained by the RC model.

---

## 1. Background

### 1.1 The 3R2C Model

The building thermal zone is approximated as a **3R2C network** consisting of:
- Indoor air capacitance (Cᵢₙ),
- Envelope capacitance (Cₑₙ),
- Resistances between nodes:
  - Rᵢₐ: indoor to ambient,
  - Rᵢₑ: indoor to envelope,
  - Rₑₐ: envelope to ambient.

The thermal balance equations are:

$$
C_{in} \frac{dT_{in}}{dt} =
\frac{T_e - T_{in}}{R_{ie}} +
\frac{T_a - T_{in}}{R_{ia}} +
a_{\text{sol,in}} Q_{\text{sol}} +
a_{\text{int,in}} Q_{\text{int}} +
Q_{ah}
$$

$$
C_{en} \frac{dT_e}{dt} =
\frac{T_{in} - T_e}{R_{ie}} +
\frac{T_a - T_e}{R_{ea}} +
a_{\text{sol,en}} Q_{\text{sol}} +
a_{\text{int,en}} Q_{\text{int}}
$$

where:
- Tᵢₙ: indoor air temperature,
- Tₑ: envelope temperature,
- Tₐ: ambient (outdoor) temperature,
- Qₛₒₗ: solar gains (irradiance × glazing area × transmittance),
- Qᵢₙₜ: internal gains,
- Qₐₕ: HVAC/air handling unit heat flux,
- aₛₒₗ,ᵢₙ, aₛₒₗ,ₑₙ, aᵢₙₜ,ᵢₙ, aᵢₙₜ,ₑₙ: absorption distribution coefficients.

---

### 1.2 Discretisation: Backward Euler

The state vector is:

$$
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
$$

The continuous dynamics can be written as:

$$
\dot{x}(t) = A x(t) + B u(t)
$$

Backward Euler integration gives a stable discrete-time update:

$$
(I - \Delta t A) x_{k+1} = x_k + \Delta t \, (B u_{k+1})
$$

where:
- Δt is the timestep (30 minutes in our dataset),
- A, B are system matrices derived from resistances and capacitances.

---

## 2. Bayesian Calibration with ADVI

Initial parameter estimates are treated as **priors**.  
These were calculated as analytical approximations for each home.
In unconstrained space (ℝ):

- Positive-only parameters (R, C, σ) use an exponential transform:

$$
z \sim \mathcal{N}(\mu, \sigma^2), \quad R = \exp(z)
$$

- Fractional parameters (a ∈ (0,1)) use the sigmoid transform:
  
$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

ADVI fits the posterior by maximising the **Evidence Lower Bound (ELBO):**

$$
\text{ELBO} = \mathbb{E}_q[\log p(D \mid \theta)] - \text{KL}(q(\theta) \| p(\theta))
$$

This yields calibrated RC parameters with uncertainty estimates.

---

## 3. Residual Learning with Bayesian LSTM

The RC model alone cannot capture all dynamics (e.g. sensor noise, measurement error, or unmodelled thermal dynamics).  
We define the **residual heating/cooling rate** as:

$$
y_t = \frac{\Delta T_{in,t}}{\Delta t} - \frac{\Delta T_{in,t}^{RC}}{\Delta t}
$$

where:
- ΔTᵢₙ,ₜ: observed indoor temperature change,
- ΔTᵢₙ,ₜᴿᶜ: RC-predicted change.

### LSTM Residual Model

An LSTM receives a **48-step input sequence (24 hours)** of features:
- Weather: Tₐ, irradiance, solar angles,
- HVAC/internal gains,
- RC predictions,
- Cyclical encodings of hour-of-day and day-of-week.

It outputs a predictive distribution over yₜ.

### Bayesian Output Layer

A **Bayesian linear layer** provides uncertainty-aware predictions:

$$
\hat{y}_t \sim \mathcal{N}(\mu_t, \sigma_t^2)
$$

Weights are treated as random variables with Gaussian priors.  
The loss combines:
- Gaussian negative log-likelihood (NLL),
- KL-divergence between approximate posterior and prior.

---

## 4. Monte Carlo Prediction

At test time, uncertainty is quantified via **Monte Carlo sampling**:
1. Draw M stochastic forward passes through the Bayesian LSTM.
2. Integrate predicted residuals into the RC baseline:

$$
T_{in,t}^{\text{pred}} = T_{in,t}^{RC} + r_t
$$

3. Collect ensemble statistics (mean, standard deviation) to form **predictive bands**.

---

## 5. Metrics

Performance is evaluated using:
- **RMSE** (Root Mean Squared Error):
  
$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left( T_{in,t}^{\text{pred}} - T_{in,t}^{\text{true}} \right)^2 }
$$

- **CVRMSE** (Coefficient of Variation of RMSE):
  
$$
\text{CVRMSE} = 100 \times \frac{\text{RMSE}}{\bar{T}_{in}}
$$

---

## 6. Preprocessed Data

While not provided here (for data protection), this model requires time-aligned inputs:
- Heat pump output power [W],
- Irradiance [W/m²], glazing area [m²], and transmittance (for solar gain),
- Internal gain [W] (e.g. occupancy schedules per ISO standards),
- Fabric resistance [K/W] (from thermal transmittance of the envelope),
- Ventilation resistance [K/W] (from Home Energy Model methodology, dynamic value),
- Solar elevation/azimuth angles (LSTM only).

---

## 7. Repository Structure

- `3R2C.py`: Physics-only RC model + ADVI calibration.
- `RC-LSTM.py`: Hybrid RC–LSTM implementation.
- `README.md`: This document.

---

## 8. Usage

```bash
# Train RC model parameters with ADVI
python 3R2C.py

# Train hybrid RC–LSTM model
python RC-LSTM.py
