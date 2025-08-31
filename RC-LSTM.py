# ==============================================================
# Hybrid RC–LSTM Model for Indoor Temperature Forecasting
# --------------------------------------------------------------
# Step 1: RC model (physics-based) with ADVI calibration.
# Step 2: LSTM learns residual dynamics to capture unmodelled effects.
# Step 3: Bayesian head provides predictive uncertainty.
#
# Author: [Your Name]
# ==============================================================

# ==== 1) Imports & Global Config ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Precision & device
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths & columns
CSV_PATH = "your_data.csv"   # <-- set to your processed dataset
TIME_COL = "timestamp"
COLS = dict(
    Tin="ble_temp", To="outdoor_temp", I="POA_total",
    Qint="internal_gain", Qah="heat_flux", Ria="R_ia",
    el="solar_elevation_deg", az="solar_azimuth_deg"
)

# Physical constants
G_TRANSMITTANCE = 0.76
A_Z_FIXED       = 41.3

# Hyperparameters
ADVI_ITERS = 3000
ADVI_LR    = 1e-2
ADVI_SEED  = 42
SEQ_LEN    = 48
LSTM_HIDDEN= 96
DROPOUT    = 0.2
BATCH      = 64
LR         = 1e-3
SEED       = 42
HEAD_PRIOR_STD = 1e-3
BETA_EPOCHS    = 40
TEST_DAYS = 28

np.random.seed(SEED); torch.manual_seed(SEED)


# ==== 2) Load Data & Features ===================================
df = pd.read_csv(CSV_PATH)
df["time"] = pd.to_datetime(df[TIME_COL], errors="coerce", dayfirst=True)
df = df.sort_values("time").set_index("time").asfreq("30T")

# Ensure numeric
for c in COLS.values():
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Clamp/clean
df[COLS["I"]]  = df[COLS["I"]].clip(lower=0)
df[COLS["el"]] = df[COLS["el"]].clip(-5, 95)
df[COLS["az"]] = df[COLS["az"]].clip(-360, 360)

# Splits
t0         = df.index.min()
advi_end   = t0 + pd.Timedelta(days=7)
test_start = df.index.max() - pd.Timedelta(days=TEST_DAYS)
mask_advi  = (df.index >= t0) & (df.index < advi_end)
mask_te    = (df.index >= test_start)

dt_s = (df.index[1] - df.index[0]).total_seconds()
DT_H = dt_s / 3600.0

# Arrays
Tin   = df[COLS["Tin"]].to_numpy(np.float32)
To    = df[COLS["To"]].to_numpy(np.float32)
Irr   = df[COLS["I"]].to_numpy(np.float32)
Qint  = df[COLS["Qint"]].to_numpy(np.float32)
Qah   = df[COLS["Qah"]].to_numpy(np.float32)
Ria   = df[COLS["Ria"]].to_numpy(np.float32)
el    = np.deg2rad(df[COLS["el"]].to_numpy(np.float32))
az    = np.deg2rad(df[COLS["az"]].to_numpy(np.float32))

# Cyclical encodings
hours = (df.index.view("int64") // 10**9) % (24*3600) / 3600.0
h_sin, h_cos = np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)
dow = df.index.weekday.values
dow_sin, dow_cos = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)

# Solar trig
el_sin, el_cos = np.sin(el), np.cos(el)
az_sin, az_cos = np.sin(az), np.cos(az)


# ==== 3) RC Model ===============================================
class RCBackwardEuler(nn.Module):
    """2R2C thermal network solved via backward Euler integration."""
    def __init__(self, dt_s, g=0.76, Az=41.3):
        super().__init__()
        self.dt = torch.tensor(dt_s, dtype=DTYPE, device=DEVICE)
        self.g  = torch.tensor(g,   dtype=DTYPE, device=DEVICE)
        self.Az = torch.tensor(Az,  dtype=DTYPE, device=DEVICE)
    def forward(self, params, Tin0, To, Irr, Qint, Qah, Ria):
        n = To.shape[0]
        Te, Tin = Tin0.to(DTYPE), Tin0.to(DTYPE)
        out = []
        I2 = torch.eye(2, dtype=DTYPE, device=DEVICE)
        R_ie,R_ea,C_in,C_en = (params["R_ie"], params["R_ea"],
                               params["C_in"], params["C_en"])
        a_si,a_se,a_ii,a_ie = (params["a_sol_in"], params["a_sol_en"],
                               params["a_int_in"], params["a_int_en"])
        for k in range(n):
            Ria_k = torch.clamp(Ria[k], min=1e-6)
            A11 = -(1.0/R_ea + 1.0/R_ie)/C_en
            A12 =  (1.0/R_ie)/C_en
            A21 =  (1.0/R_ie)/C_in
            A22 = -(1.0/Ria_k + 1.0/R_ie)/C_in
            A = torch.stack([torch.stack([A11,A12]), torch.stack([A21,A22])])
            B = torch.stack([
                torch.stack([1.0/(C_en*R_ea), a_se/C_en, a_ie/C_en, torch.tensor(0., dtype=DTYPE, device=DEVICE)]),
                torch.stack([1.0/(C_in*Ria_k), a_si/C_in, a_ii/C_in, 1.0/C_in])
            ])
            Qsol = self.g * self.Az * Irr[k]
            u = torch.stack([To[k], Qsol, Qint[k], Qah[k]])
            x   = torch.stack([Te, Tin])
            M   = I2 - self.dt*A
            rhs = x + self.dt*(B@u)
            x_new = torch.linalg.solve(M, rhs)
            Te, Tin = x_new[0], x_new[1]
            out.append(Tin)
        return torch.stack(out)


# ==== 4) ADVI Calibration ========================================
# (For brevity: reuse your advi_rc_week1() function from rc_model.py)
# Posterior mean parameters (P_post) are used below.


# ==== 5) RC Rollout + Residuals ==================================
def rc_rollout(params_nat, Tin_np, To_np, Irr_np, Qint_np, Qah_np, Ria_np):
    """Rollout RC model with fixed parameters."""
    rc = RCBackwardEuler(dt_s, G_TRANSMITTANCE, A_Z_FIXED).to(DEVICE)
    params_t = {k: torch.tensor(v, dtype=DTYPE, device=DEVICE) for k,v in params_nat.items()}
    with torch.no_grad():
        out = rc(params_t,
                 torch.tensor(float(Tin_np[0]), dtype=DTYPE, device=DEVICE),
                 torch.tensor(To_np,   dtype=DTYPE, device=DEVICE),
                 torch.tensor(Irr_np,  dtype=DTYPE, device=DEVICE),
                 torch.tensor(Qint_np, dtype=DTYPE, device=DEVICE),
                 torch.tensor(Qah_np,  dtype=DTYPE, device=DEVICE),
                 torch.tensor(Ria_np,  dtype=DTYPE, device=DEVICE)).cpu().numpy()
    return out


# ==== 6) Dataset for LSTM ========================================
class SeqDSSingle(Dataset):
    """Sliding-window dataset for residual ΔT dynamics."""
    def __init__(self, Xn, y, mask, seq_len):
        self.Xn, self.y, self.seq_len = Xn, y.astype(np.float32), seq_len
        idx = np.where(mask)[0]
        ends = []
        for e in idx:
            s = e - (seq_len-1)
            if s >= 0 and np.isfinite(Xn[s:e+1]).all() and np.isfinite(y[e]):
                ends.append(e)
        self.ends = np.asarray(ends, dtype=int)
    def __len__(self): return len(self.ends)
    def __getitem__(self, i):
        e = self.ends[i]; s = e-(self.seq_len-1)
        return (torch.tensor(self.Xn[s:e+1], dtype=DTYPE),
                torch.tensor([self.y[e]], dtype=DTYPE),
                int(e))


# ==== 7) Bayesian LSTM ===========================================
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1e-3):
        super().__init__()
        self.w_mu  = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), -3.0))
        self.register_buffer("prior_var", torch.tensor(prior_std**2, dtype=DTYPE))
    def forward(self, x):
        w = self.w_mu + F.softplus(self.w_rho)*torch.randn_like(self.w_mu)
        b = self.b_mu + F.softplus(self.b_rho)*torch.randn_like(self.b_mu)
        return x@w.T + b
    def kl(self):
        def kl_gauss(mu,rho):
            std = F.softplus(rho); var=std**2
            return 0.5*torch.sum((var+mu**2)/self.prior_var - 1.0 +
                                 (torch.log(self.prior_var)-torch.log(var+1e-12)))
        return kl_gauss(self.w_mu,self.w_rho)+kl_gauss(self.b_mu,self.b_rho)

class ResidualLSTM_Bayes(nn.Module):
    def __init__(self, in_dim, hidden=96, dropout=0.2, prior_std=1e-3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head_mu = BayesianLinear(hidden, 1, prior_std=prior_std)
        self.head_logvar = nn.Linear(hidden, 1)
        self.logvar_bias = nn.Parameter(torch.tensor(-0.5, dtype=DTYPE))
    def forward(self,x):
        h,_ = self.lstm(x); h=self.drop(h[:,-1,:])
        mu = self.head_mu(h)
        logvar = torch.clamp(self.head_logvar(h)+self.logvar_bias,-8,6)
        return mu,logvar
    def kl(self): return self.head_mu.kl()


# ==== 8) Training =================================================
def nll_gauss(y, mu, logvar):
    return 0.5*torch.mean((y-mu)**2*torch.exp(-logvar)+logvar)

def run_epoch(loader, train=True, epoch=1):
    model.train(train)
    total, nobs = 0.0,0
    beta = min(1.0, epoch/max(1,BETA_EPOCHS))
    for Xw,y1,_ in loader:
        Xw,y1 = Xw.to(DEVICE),y1.to(DEVICE)
        mu,logvar = model(Xw)
        nll = nll_gauss(y1,mu,logvar)
        kl  = model.kl()/max(1,Xw.size(0))
        loss = nll + beta*kl
        if train:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        total += loss.item()*Xw.size(0); nobs+=Xw.size(0)
    return total/max(1,nobs)


# ==== 9) Monte Carlo Prediction ==================================
def mc_predict_level(ds, M=200):
    """Monte Carlo integration of Δ-residuals to predict Tin."""
    model.train(True)
    Xcat, ends=[], ds.ends
    for xb,yb,_ in DataLoader(ds,batch_size=256,shuffle=False):
        Xcat.append(xb)
    Xcat=torch.cat(Xcat,0).to(DEVICE)
    r_prev = res_full[:-1][ends]
    T_rc1  = Tin_rc[1:][ends]
    T_true = Tin[1:][ends]
    samples=[]
    with torch.no_grad():
        for _ in range(M):
            mu,logvar=model(Xcat)
            y_s=(mu+torch.exp(0.5*logvar)*torch.randn_like(mu))[:,0].cpu().numpy()
            r_next=r_prev+y_s*DT_H
            T_pred=T_rc1+r_next
            samples.append(T_pred)
    S=np.stack(samples,0)
    return S.mean(axis=0),S.std(axis=0),T_true,T_rc1,ends


# ==== 10) Metrics ================================================
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def cvrmse(a,b): return 100.0*rmse(a,b)/float(np.mean(b))
