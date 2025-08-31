# 0) Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assign single precision for computational speed
DTYPE  = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(DTYPE)

# Input Data
INPUT_FILE  = "Data.csv"

# Column mapping
COLS = dict(
    Tin="indoor_temp",
    To="outdoor_temp",
    I="solar_total",
    Qint="internal_gain",
    Qah="heat_flux",
    Ria="ventilation_res",
)

# Transmittance (g value) for windows
G_TRANSMITTANCE = 0.76
# Total glazed area (m^2)
A_Z_FIXED       = 41.3

# Training/Testing Splits
TEST_DAYS   = 28
ADVI_DAYS   = 7
# Adjust as required, usually convergence achieved within ~2000 iterations
ADVI_ITERS  = 3000
# Learning rate for Adam optimiser
ADVI_LR     = 1e-2
ADVI_SEED   = 42
# Gardually increase the KL penalty so that it doesn't defect to the prior
KL_WARMUP   = 800

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# 1) Load preprocessed data
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

# Required input columns (assumed pre-cleaned and aligned to 30 min)
Tin  = df[COLS["Tin"]].to_numpy(np.float32)
To   = df[COLS["To"]].to_numpy(np.float32)
Irr  = df[COLS["I"]].to_numpy(np.float32)
Qint = df[COLS["Qint"]].to_numpy(np.float32)
Qah  = df[COLS["Qah"]].to_numpy(np.float32)
Ria  = df[COLS["Ria"]].to_numpy(np.float32)

# Timesteps are in 30 minute intervals (1800 s)
dt_s = 1800.0
DT_H = dt_s / 3600.0

# Define ADVI training window (first 7 days of data)
steps_week = int(pd.Timedelta(days=ADVI_DAYS).total_seconds() // dt_s)
mask_advi  = np.arange(steps_week)


class RCBackwardEuler(nn.Module):
    """
    3R2C thermal network solved with backward Euler integration.
    States:
        Te  = envelope temperature
        Tin = indoor air temperature
    Inputs:
        To   = outdoor temperature
        Irr  = solar irradiance
        Qint = internal gains
        Qah  = ASHP heat flux (input heating/cooling)
        Ria  = ventilation resistance
    """

    def __init__(self, dt_s, g=G_TRANSMITTANCE, Az=A_Z_FIXED, substeps=1):
        super().__init__()
        # Time step (s), solar transmittance, glazed area
        self.dt = torch.tensor(dt_s, dtype=DTYPE, device=DEVICE)
        self.g  = torch.tensor(g,   dtype=DTYPE, device=DEVICE)
        self.Az = torch.tensor(Az,  dtype=DTYPE, device=DEVICE)
        self.substeps = int(substeps)
        
    def forward(self, params, Tin0, To, Irr, Qint, Qah, Ria):
        """
        Roll out indoor temperature trajectory given parameters and inputs.
        params : dict of RC parameters (resistances, capacitances, absorption coefficients)
        Tin0   : initial indoor temperature
        Returns: torch.Tensor of predicted indoor temperatures
        """
        n = To.shape[0]                  # number of time steps
        Te = Tin0.to(DTYPE)              # initial envelope temperature
        Tin = Tin0.to(DTYPE)             # initial indoor temperature
        out = []                         # store outputs
        I2 = torch.eye(2, dtype=DTYPE, device=DEVICE)  # 2x2 identity matrix

        # Extract RC parameters
        R_ie,R_ea,C_in,C_en = (params["R_ie"].to(DTYPE),
                               params["R_ea"].to(DTYPE),
                               params["C_in"].to(DTYPE),
                               params["C_en"].to(DTYPE))
        # Absorption coefficients for solar/internal splits
        a_si,a_se,a_ii,a_ie = (params["a_sol_in"].to(DTYPE),
                               params["a_sol_en"].to(DTYPE),
                               params["a_int_in"].to(DTYPE),
                               params["a_int_en"].to(DTYPE))

        # Substep size (for stability if needed)
        sub = max(1, self.substeps)
        dt_sub = self.dt / sub

        # Time-stepping loop
        for k in range(n):
            # Place lower bound on ventilation resistance to avoid division by zero
            Ria_k = torch.clamp(Ria[k].to(DTYPE), min=1e-4)

            # System dynamics matrix A (state space)
            A11 = -(1.0/R_ea + 1.0/R_ie)/C_en
            A12 =  (1.0/R_ie)/C_en
            A21 =  (1.0/R_ie)/C_in
            A22 = -(1.0/Ria_k + 1.0/R_ie)/C_in
            A = torch.stack([torch.stack([A11, A12]), torch.stack([A21, A22])])

            # Input coupling matrix B (state space)
            B = torch.stack([
                torch.stack([1.0/(C_en*R_ea), a_se/C_en, a_ie/C_en, torch.tensor(0., dtype=DTYPE, device=DEVICE)]),
                torch.stack([1.0/(C_in*Ria_k), a_si/C_in, a_ii/C_in, 1.0/C_in])
            ])

            # Multiply irradiance by g-value and glazed area to compute solar gain
            Qsol = self.g * self.Az * torch.clamp(Irr[k].to(DTYPE), min=0., max=2000.)
            qah  = torch.clamp(Qah[k].to(DTYPE), min=0.)
            u = torch.stack([To[k].to(DTYPE), Qsol, Qint[k].to(DTYPE), qah])

            # State vector (envelope and indoor temperature)
            x = torch.stack([Te, Tin])
            # Backward-Euler matrix rearrangement
            M = I2 - dt_sub * A

            # Backward Euler integration (for stability, substepping if needed)
            for _ in range(sub):
                rhs = x + dt_sub * (B @ u)
                # Solve linear system
                x   = torch.linalg.solve(M, rhs)

            Te, Tin = x[0], x[1]
            # Clamp Tin to realistic comfort range
            Tin = torch.clamp(Tin, min=5., max=45.)
            out.append(Tin)

        return torch.stack(out)

# 3) ADVI for RC on Week-1
# Logit function is the inverse of the sigmoid (unconstrained -> constrained)
def logit(p): return np.log(p/(1-p))

# Assign prior values from analytical calculations
ADVI_PRIORS = dict(
    # Resistance / Capacitance
    R_ie     = dict(mean=0.0094595,  sd_uncon=1.0), # K/W
    R_ea     = dict(mean=0.0094595,  sd_uncon=1.0), # K/W
    C_in     = dict(mean=317_137.8,  sd_uncon=1.0), # J/K
    C_en     = dict(mean=55_500_000, sd_uncon=1.0), # J/K
    # Absorption Coefficients
    a_sol_in = dict(mean=0.5,        sd_uncon=0.7), # More confidence for coefficients (lower sd)
    a_sol_en = dict(mean=0.5,        sd_uncon=0.7),
    a_int_in = dict(mean=0.5,        sd_uncon=0.7),
    a_int_en = dict(mean=0.5,        sd_uncon=0.7),
    # Standard deviation of the residual noise between model predictions and observed indoor temperatures.
    sigma    = dict(mean=1.5,        sd_uncon=0.6), 
)
# Define ADVI function, "tr_np" = NumPy array of training data
def advi_rc_week1(
    Tin_tr_np, To_tr_np, Irr_tr_np, Qint_tr_np, Qah_tr_np, Ria_tr_np,
    # Initialise previously defined parameters
    dt_s, iters=ADVI_ITERS, lr=ADVI_LR, kl_warmup_iters=KL_WARMUP, seed=ADVI_SEED,
    # Bounds on standard deviation noise
    sigma_bounds=(1e-3, 50.0)
):
    torch.manual_seed(seed)
    rc = RCBackwardEuler(dt_s, G_TRANSMITTANCE, A_Z_FIXED).to(DEVICE)
    # Convert training data to tensors
    Tin_tr = torch.tensor(Tin_tr_np,  dtype=DTYPE, device=DEVICE)
    To_tr  = torch.tensor(To_tr_np,   dtype=DTYPE, device=DEVICE)
    Irr_tr = torch.tensor(Irr_tr_np,  dtype=DTYPE, device=DEVICE)
    Qint_tr= torch.tensor(Qint_tr_np, dtype=DTYPE, device=DEVICE)
    Qah_tr = torch.tensor(Qah_tr_np,  dtype=DTYPE, device=DEVICE)
    Ria_tr = torch.tensor(Ria_tr_np,  dtype=DTYPE, device=DEVICE)
    # Initial condition
    Tin0   = Tin_tr[0]

    # Debugger to check for NaNs / infinities in the ADVI train window
    for name,t in dict(Tin=Tin_tr,To=To_tr,Irr=Irr_tr,Qint=Qint_tr,Qah=Qah_tr,Ria=Ria_tr).items():
        if not torch.isfinite(t).all():
            raise ValueError(f"[ADVI] Non-finite values in Week-1 {name}")

    # Variational parameter constructor in the unconstrained space:
    #   - 'mu' is the mean of q(z)
    #   - 'rho' parameterizes std via softplus(rho) to ensure positivity
    def make_var_uncon(mu0):
        mu  = torch.tensor(mu0, dtype=DTYPE, device=DEVICE, requires_grad=True)
        rho = torch.tensor(-2.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
        return dict(mu=mu, rho=rho)

    # Build Gaussian priors in the unconstrained space:
    #   - positive params -> log-transform
    #   - (0,1) params   -> logit-transform
    #   - sigma          -> log-transform (stored as "log_sigma")
    prior_mu, prior_sd = {}, {}
    for name in ["R_ie","R_ea","C_in","C_en"]:
        m = ADVI_PRIORS[name]["mean"]
        prior_mu[name] = torch.tensor(np.log(m), dtype=DTYPE, device=DEVICE)
        prior_sd[name] = torch.tensor(ADVI_PRIORS[name]["sd_uncon"], dtype=DTYPE, device=DEVICE)
    for name in ["a_sol_in","a_sol_en","a_int_in","a_int_en"]:
        m = ADVI_PRIORS[name]["mean"]
        prior_mu[name] = torch.tensor(logit(m), dtype=DTYPE, device=DEVICE)
        prior_sd[name] = torch.tensor(ADVI_PRIORS[name]["sd_uncon"], dtype=DTYPE, device=DEVICE)
    prior_mu["log_sigma"] = torch.tensor(np.log(ADVI_PRIORS["sigma"]["mean"]), dtype=DTYPE, device=DEVICE)
    prior_sd["log_sigma"] = torch.tensor(ADVI_PRIORS["sigma"]["sd_uncon"],   dtype=DTYPE, device=DEVICE)
    
    # Variational family q(z) = N(mu, softplus(rho)^2) in the unconstrained space for each parameter
    q = {
        "R_ie": make_var_uncon(prior_mu["R_ie"].item()),
        "R_ea": make_var_uncon(prior_mu["R_ea"].item()),
        "C_in": make_var_uncon(prior_mu["C_in"].item()),
        "C_en": make_var_uncon(prior_mu["C_en"].item()),
        "a_sol_in": make_var_uncon(prior_mu["a_sol_in"].item()),
        "a_sol_en": make_var_uncon(prior_mu["a_sol_en"].item()),
        "a_int_in": make_var_uncon(prior_mu["a_int_in"].item()),
        "a_int_en": make_var_uncon(prior_mu["a_int_en"].item()),
        "log_sigma": make_var_uncon(prior_mu["log_sigma"].item()),
    }
    # Adam optimiser over all variational parameters (both mu and rho for every entry in q)
    opt = torch.optim.Adam([p for d in q.values() for p in d.values()], lr=lr)

    def sample_params_and_kl(kl_scale=1.0):
        """
        Reparameterization trick:
          - sample z ~ N(mu, std^2) in unconstrained space,
          - map to constrained parameters via exp(.) / sigmoid(.),
          - accumulate analytic KL(q || p) to Gaussian priors in the same space.
        """
        params = {}; sigma = None
        kl = torch.tensor(0., dtype=DTYPE, device=DEVICE)
        for name, vr in q.items():
            mu, rho = vr["mu"], vr["rho"]
            std = F.softplus(rho)
            z   = mu + std * torch.randn_like(mu)

            # Analytic KL for 1D Gaussians in unconstrained space
            pmu, psd = prior_mu[name], prior_sd[name]
            kl_term = torch.log(psd/std) + (std**2 + (mu - pmu)**2)/(2*psd**2) - 0.5
            kl += kl_scale * kl_term

            # Map to constrained parameter spaces
            if name in ["R_ie","R_ea","C_in","C_en"]:
                params[name] = torch.exp(z)        # positive
            elif name in ["a_sol_in","a_sol_en","a_int_in","a_int_en"]:
                params[name] = torch.sigmoid(z)    # (0,1)
            elif name == "log_sigma":
                sigma = torch.exp(z).squeeze()     # positive noise std
        # Stabilise observation noise
        sigma_eff = torch.clamp(sigma, sigma_bounds[0], sigma_bounds[1])
        return params, sigma_eff, kl

    # Main ADVI loop: maximize ELBO = E_q[log p(data | params)] - KL(q || p)
    for it in range(1, iters+1):
        # KL warmup ramps prior strength from 0 -> 1 to avoid early posterior collapse
        kl_scale = min(1.0, it / max(1, kl_warmup_iters))
        opt.zero_grad()
        params, sigma_eff, kl = sample_params_and_kl(kl_scale)
        # Forward pass through the RC model (T_in_hat = prediction)
        Tin_hat = rc(params, Tin0, To_tr, Irr_tr, Qint_tr, Qah_tr, Ria_tr)
        if not torch.isfinite(Tin_hat).all():
            raise FloatingPointError("[ADVI] Tin_hat became non-finite; check inputs and R_ia range.")

        ll = -0.5*torch.sum(((Tin_tr - Tin_hat)/sigma_eff)**2 + 2*torch.log(sigma_eff) + np.log(2*np.pi))
        elbo = ll - kl
        if not torch.isfinite(elbo):
            raise FloatingPointError("[ADVI] Non-finite ELBO encountered.")
        
        # Gradient ascent on ELBO  <=>  gradient descent on (-ELBO)
        (-elbo).backward()
        torch.nn.utils.clip_grad_norm_([p for d in q.values() for p in d.values()], max_norm=5.0)
        opt.step()
        # Print every 500 iterations for convergence check
        if it % 500 == 0:
            print(f"[ADVI] iter {it:4d} | ELBO={elbo.item():.1f} | KL={kl.item():.2f} | "
                  f"sigma={float(sigma_eff):.3f} | kl_scale={kl_scale:.2f}")
            
    # Extract posterior point estimates (posterior means in unconstrained space -> map back)
    with torch.no_grad():
        P_post = dict(
            R_ie     = float(torch.exp(q["R_ie"]["mu"])),
            R_ea     = float(torch.exp(q["R_ea"]["mu"])),
            C_in     = float(torch.exp(q["C_in"]["mu"])),
            C_en     = float(torch.exp(q["C_en"]["mu"])),
            a_sol_in = float(torch.sigmoid(q["a_sol_in"]["mu"])),
            a_sol_en = float(torch.sigmoid(q["a_sol_en"]["mu"])),
            a_int_in = float(torch.sigmoid(q["a_int_in"]["mu"])),
            a_int_en = float(torch.sigmoid(q["a_int_en"]["mu"])),
        )
        # Keep full variational params for posterior sampling later (predictive bands, uncertainty)
        q_post = { n: dict(mu=q[n]["mu"].detach().clone(), rho=q[n]["rho"].detach().clone())
                   for n in q.keys() }
    return P_post, q_post

# Fit ADVI on Week-1 training segment
P_post, q_post = advi_rc_week1(
    Tin[mask_advi], To[mask_advi], Irr[mask_advi], Qint[mask_advi], Qah[mask_advi], Ria[mask_advi],
    dt_s, iters=ADVI_ITERS, lr=ADVI_LR, kl_warmup_iters=KL_WARMUP, seed=ADVI_SEED
)

print("\nPosterior mean parameters (RC):")
for k,v in P_post.items():
    print(f"  {k:10s} = {v:.6g}")

# 4) 96-hour FREE-RUN forecast on a FIXED shared window (poster style) ====
# For the sake of visual clarity we chose a 4-day forecast, this can be increased

HOURS = 96

# Start of the forecast window (chosen manually here)
FIXED_START_STR = "2024-04-04 00:00:00"
fixed_start = pd.to_datetime(FIXED_START_STR)

# Convert the fixed start into a positional slice [start:end)
# - need = number of timesteps corresponding to 96 hours
# - start_pos = index position of the fixed start
# - end_pos = start_pos + number of required steps
need = int(round(pd.Timedelta(hours=HOURS).total_seconds() / dt_s))  # e.g. 192 steps at 30-min intervals
start_pos = int(np.searchsorted(df.index.values, np.array(fixed_start, dtype="datetime64[ns]"), side="left"))
end_pos   = start_pos + need
# Ensure period doesn't extend past last datapoint
if end_pos > len(df.index):
    raise RuntimeError("Fixed 96h window extends beyond available data.")

fcst_slice = slice(start_pos, end_pos)

# Posterior sampling and rollout

def sample_rc_params_nat(q_post, S=300, include_sigma=True):
    """
    Draw S samples from the variational posterior (q_post) and transform
    them back to the natural parameter space:
      - Resistances/Capacitances: exp(z) > 0
      - Fractions (alpha): sigmoid(z), (0,1)
      - Noise sigma: exp(z) > 0
    Returns:
      pars  : list of sampled parameter dictionaries
      sigma : representative observational noise (median of draws)
    """
    pars, sigmas = [], []
    for _ in range(S):
        p = {}
        for name, vr in q_post.items():
            mu, rho = vr["mu"], vr["rho"]
            std = F.softplus(rho)             # ensures std > 0
            z   = mu + std * torch.randn_like(mu)  # reparam trick: sample z ~ N(mu, std)
            # Map back to constrained space depending on parameter type
            if name in ["R_ie","R_ea","C_in","C_en"]:
                p[name] = float(torch.exp(z).item())
            elif name in ["a_sol_in","a_sol_en","a_int_in","a_int_en"]:
                p[name] = float(torch.sigmoid(z).item())
            elif name == "log_sigma" and include_sigma:
                sigmas.append(float(torch.exp(z).item()))
        pars.append(p)
    sigma = float(np.median(sigmas)) if (include_sigma and len(sigmas) > 0) else 0.0
    return pars, sigma


def rc_rollout_free_run(params_nat, Tin_np, To_np, Irr_np, Qint_np, Qah_np, Ria_np, dt_s):
    """
    Run a forward simulation of the RC model using given parameters.
    Inputs are numpy arrays, converted to tensors.
    Returns: predicted indoor temperature trajectory as a numpy array.
    """
    rc = RCBackwardEuler(dt_s, G_TRANSMITTANCE, A_Z_FIXED).to(DEVICE)
    params_t = {k: torch.tensor(v, dtype=DTYPE, device=DEVICE) for k, v in params_nat.items()}
    To_t   = torch.tensor(To_np,   dtype=DTYPE, device=DEVICE)
    Irr_t  = torch.tensor(Irr_np,  dtype=DTYPE, device=DEVICE)
    Qint_t = torch.tensor(Qint_np, dtype=DTYPE, device=DEVICE)
    Qah_t  = torch.tensor(Qah_np,  dtype=DTYPE, device=DEVICE)
    Ria_t  = torch.tensor(Ria_np,  dtype=DTYPE, device=DEVICE)
    Tin0_t = torch.tensor(float(Tin_np[0]), dtype=DTYPE, device=DEVICE)
    with torch.no_grad():  # no gradient tracking needed
        out = rc(params_t, Tin0_t, To_t, Irr_t, Qint_t, Qah_t, Ria_t).cpu().numpy()
    return out


def rc_posterior_bands_fixed(sl, S=300, add_obs_noise=True):
    """
    Generate posterior predictive bands for a fixed forecast window.
    - Samples S parameter sets from the posterior
    - Runs free-run RC simulations
    - Adds Gaussian observation noise
    Returns:
      mean trajectory (ensemble average)
      std trajectory (ensemble spread)
    """
    Tin_sl  = Tin[sl]; To_sl   = To[sl];   Irr_sl  = Irr[sl]
    Qint_sl = Qint[sl]; Qah_sl = Qah[sl]; Ria_sl  = Ria[sl]
    draws, sigma = sample_rc_params_nat(q_post, S=S, include_sigma=True)
    sims = []
    for p in draws:
        sim = rc_rollout_free_run(p, Tin_sl, To_sl, Irr_sl, Qint_sl, Qah_sl, Ria_sl, dt_s)
        if add_obs_noise and sigma > 0:
            sim = sim + np.random.normal(0.0, sigma, size=sim.shape).astype(np.float32)
        sims.append(sim)
    Sarr = np.stack(sims, 0)
    return Sarr.mean(axis=0), Sarr.std(axis=0)


# Run posterior predictive simulation on the fixed window
mean_rc, std_rc = rc_posterior_bands_fixed(fcst_slice, S=300, add_obs_noise=True)
Tin_fc  = Tin[fcst_slice]   # truth (observed indoor temp)
idx_rc  = df.index[fcst_slice]  # timestamps

# === Error metrics ===
def rmse(a,b):
    """ Root Mean Squared Error (RMSE) between arrays """
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a-b)**2)))

def cvrmse(a,b): 
    """ Coefficient of Variation of RMSE (% of mean observed value) """
    return 100.0*rmse(a,b)/float(np.mean(b))

print(f"[RC fixed {HOURS}h] RMSE={rmse(mean_rc, Tin_fc):.2f} °C  |  CVRMSE={cvrmse(mean_rc, Tin_fc):.2f}%")