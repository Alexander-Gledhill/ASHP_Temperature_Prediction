# Hybrid RC–LSTM Model for Indoor Temperature Forecasting
# Step 1: RC model (physics-based) with ADVI calibration (see 3R2C code for full details).
# Step 2: LSTM learns residual dynamics to capture unmodelled effects.
# Step 3: Bayesian head provides predictive uncertainty.

# 1) Imports and Configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Single precision for computational speed
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths & columns
CSV_PATH = "your_data.csv"   # preprocessed dataset
TIME_COL = "timestamp"
COLS = dict(
    Tin="indoor_temp", To="outdoor_temp", I="total_irradiance",
    Qint="internal_gain", Qah="heat_flux", Ria="R_ia",
    el="solar_elevation_deg", az="solar_azimuth_deg"        # Note that the LSTM uses elevation/azimuth sun angle as inputs
)

# Window transmittance (g-value) and glazed area (m^2)
G_TRANSMITTANCE = 0.76
A_Z_FIXED       = 41.3

# Hyperparameters
ADVI_ITERS = 3000    # Maximum iterations, 3000 often unnecessary, 2000 okay
ADVI_LR    = 1e-2    # Learning rate for Adam optimiser (balance for convergence)
ADVI_SEED  = 42
SEQ_LEN    = 48    # Input window length, e.g. number of past time steps to predict next residual (24 hrs)
LSTM_HIDDEN= 96    # Hidden state vector
DROPOUT    = 0.2    # Monte Carlo Dropout
BATCH      = 64
LR         = 1e-3    # Learning rate for Adam optimiser
SEED       = 42
HEAD_PRIOR_STD = 1e-3    # Prior variance on the weights of the Bayesian head
BETA_EPOCHS    = 40    # KL scaling factor increased from 0 to 1 over 40 passes
TEST_DAYS = 28

np.random.seed(SEED); torch.manual_seed(SEED)


# 2) Load Data & Features
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

# Splits for ADVI and testing
t0         = df.index.min()
advi_end   = t0 + pd.Timedelta(days=7)
test_start = df.index.max() - pd.Timedelta(days=TEST_DAYS)
mask_advi  = (df.index >= t0) & (df.index < advi_end)
mask_te    = (df.index >= test_start)

dt_s = (df.index[1] - df.index[0]).total_seconds()
DT_H = dt_s / 3600.0

# NumPy arrays for each input
Tin   = df[COLS["Tin"]].to_numpy(np.float32)
To    = df[COLS["To"]].to_numpy(np.float32)
Irr   = df[COLS["I"]].to_numpy(np.float32)
Qint  = df[COLS["Qint"]].to_numpy(np.float32)
Qah   = df[COLS["Qah"]].to_numpy(np.float32)
Ria   = df[COLS["Ria"]].to_numpy(np.float32)
el    = np.deg2rad(df[COLS["el"]].to_numpy(np.float32))
az    = np.deg2rad(df[COLS["az"]].to_numpy(np.float32))

# Cyclical encodings - needed for neural network to understand time
# Encode hours so that model understands 23:00 and 01:00 are "close" rather than far apart.
# Map time onto a circle with sine/cosine
hours = (df.index.view("int64") // 10**9) % (24*3600) / 3600.0
# e.g. 13.5 = 1.30pm
h_sin, h_cos = np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)
# maps hours onto the unit circle
dow = df.index.weekday.values
# index day of the week to 0->6.
dow_sin, dow_cos = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)

# Sine/cosine of elevation/azimuth angles
el_sin, el_cos = np.sin(el), np.cos(el)
az_sin, az_cos = np.sin(az), np.cos(az)

# 3) RC Model
class RCBackwardEuler(nn.Module):
    # 2R2C thermal network solved via Backward Euler integration
    # See "3R2C.py" for full comments
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


# 4) ADVI Calibration
# Import the ADVI function from the RC-only module (3R2C.py)
from _3R2C import advi_rc_week1  
# Run ADVI on Week-1 slice (posterior calibration of RC parameters)
P_post, q_post = advi_rc_week1(
    Tin[mask_advi], To[mask_advi], Irr[mask_advi],
    Qint[mask_advi], Qah[mask_advi], Ria[mask_advi],
    dt_s, iters=ADVI_ITERS, lr=ADVI_LR, seed=ADVI_SEED
)
# Return parameters
print("\nPosterior mean parameters (RC model):")
for k, v in P_post.items():
    print(f"  {k:10s} = {v:.6g}")


# 5) RC Rollout + Residuals
def rc_rollout(params_nat, Tin_np, To_np, Irr_np, Qint_np, Qah_np, Ria_np):
    # Rollout RC model with fixed parameters
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


# 6) Dataset for LSTM
# Allows LSTM to learn from last 24 hours of data what residual correction does the RC model need at time t
class SeqDSSingle(Dataset):
    # Sliding-window dataset for residual detla T dynamics
    def __init__(self, Xn, y, mask, seq_len):
        # Xn  = feature matrix, with N timesteps and D features
        # y = target vector (delta T residual per step)
        self.Xn, self.y, self.seq_len = Xn, y.astype(np.float32), seq_len
        idx = np.where(mask)[0] # all time indicies
        ends = []
        for e in idx:
            s = e - (seq_len-1)
            if s >= 0 and np.isfinite(Xn[s:e+1]).all() and np.isfinite(y[e]): # no NaNs in features, end feature valid
                ends.append(e)
        self.ends = np.asarray(ends, dtype=int) # list of valid endpoints where a full training window can be constructed
    def __len__(self): return len(self.ends) # number of training samples (no. of valid window endpoints)
    def __getitem__(self, i):
        e = self.ends[i]; s = e-(self.seq_len-1)    # for sample index i, get end index e, compute start s
        return (torch.tensor(self.Xn[s:e+1], dtype=DTYPE),    # extract feature matrix (Xn[s:e+1])
                torch.tensor([self.y[e]], dtype=DTYPE),    # target, a scalar delta T at time e
                int(e))


# 7) Bayesian LSTM
# Bayesian linear layer where weights and biases are treated as random variables with distributions
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1e-3):
        super().__init__()
        self.w_mu  = nn.Parameter(torch.zeros(out_features, in_features))    # Mean
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))    # Scale parameter
        self.b_mu  = nn.Parameter(torch.zeros(out_features))    # Mean
        self.b_rho = nn.Parameter(torch.full((out_features,), -3.0))    # -3 initialisation, softplus(-3) = 0.05 (e.g. small initial variance)
        self.register_buffer("prior_var", torch.tensor(prior_std**2, dtype=DTYPE))    #  Bayesian regulariser, pushes weights toward 0 unless evidence from data pulls away
    def forward(self, x):
        w = self.w_mu + F.softplus(self.w_rho)*torch.randn_like(self.w_mu)    # enables stochastic predictions, allows uncertainty estimation later
        b = self.b_mu + F.softplus(self.b_rho)*torch.randn_like(self.b_mu)
        return x@w.T + b
    def kl(self):
        def kl_gauss(mu,rho):
            # Computes Kullback-Leibler divergence between approx. posterior (q(w)) and prior p(w)
            std = F.softplus(rho); var=std**2
            return 0.5*torch.sum((var+mu**2)/self.prior_var - 1.0 +
                                 (torch.log(self.prior_var)-torch.log(var+1e-12)))
        return kl_gauss(self.w_mu,self.w_rho)+kl_gauss(self.b_mu,self.b_rho)

class ResidualLSTM_Bayes(nn.Module):
    # defines the hybrid residual Bayesian LSTM model
    # in_dim = number of input features per timestep
    # others are hyperparameters initialised at beginning
    def __init__(self, in_dim, hidden=96, dropout=0.2, prior_std=1e-3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)    # processes sequences of inuput windows
        self.drop = nn.Dropout(dropout)    # MC dropout during training and inference (stochasticity)
        self.head_mu = BayesianLinear(hidden, 1, prior_std=prior_std)    # predicts mean of delta T distribution
        self.head_logvar = nn.Linear(hidden, 1)
        self.logvar_bias = nn.Parameter(torch.tensor(-0.5, dtype=DTYPE))
    def forward(self,x):
        h,_ = self.lstm(x)     # pass sequence x into LSTM, get hidden states for each timestep
        h=self.drop(h[:,-1,:])    # keep last hidden state
        mu = self.head_mu(h)    # mu = mean of delta T_in    
        logvar = torch.clamp(self.head_logvar(h)+self.logvar_bias,-8,6)    # predicive variance
        return mu,logvar
    def kl(self): return self.head_mu.kl()    # returns KL divergence penalty, (ELBO = log-likelihood - KL)


# 8) Training
def nll_gauss(y, mu, logvar):    # logvar = log(sigma^2), exp(-logvar) = 1/sigma^2
    return 0.5*torch.mean((y-mu)**2*torch.exp(-logvar)+logvar)

def run_epoch(loader, train=True, epoch=1):    # training loop for each epoch
    model.train(train)    # training mode with dropout
    total, nobs = 0.0,0
    beta = min(1.0, epoch/max(1,BETA_EPOCHS))
    for Xw,y1,_ in loader:    # iterate over mini-samples from data loader
        Xw,y1 = Xw.to(DEVICE),y1.to(DEVICE)
        mu,logvar = model(Xw)    # outputs, mu = mean prediction, logvar = log variance prediction
        nll = nll_gauss(y1,mu,logvar)    # Gaussian negative log-likelihood
        kl  = model.kl()/max(1,Xw.size(0))    # KL divergence between posterior and prior
        loss = nll + beta*kl
        if train:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        total += loss.item()*Xw.size(0); nobs+=Xw.size(0)
    return total/max(1,nobs)
# Function above runs for one full pass (epoch) of training
# Early epochs prioritise fitting data, low beta
# Later epochs balance data fit + regularisation

# 9) Monte Carlo Prediction
# Takes dataset (ds) and 200 Monte Carlo samples to predict T_in wiht uncertainty bands
def mc_predict_level(ds, M=200):
    # Monte Carlo integration of Δ-residuals to predict Tin
    model.train(True)
    Xcat, ends=[], ds.ends    # Collects all input sequences into one larger tensor Xcat
    for xb,yb,_ in DataLoader(ds,batch_size=256,shuffle=False):
        Xcat.append(xb)
    Xcat=torch.cat(Xcat,0).to(DEVICE)
    r_prev = res_full[:-1][ends]    # residual between measured T_in and RC-predicted T_in over full dataset
    T_rc1  = Tin_rc[1:][ends]    # RC model prediction at t+1
    T_true = Tin[1:][ends]    # Measured indoor temperature at t+1 (for evaluation)
    samples=[]
    with torch.no_grad():
        for i in range(M):
            mu,logvar=model(Xcat)    # call mean and log variance
            y_s=(mu+torch.exp(0.5*logvar)*torch.randn_like(mu))[:,0].cpu().numpy()    # sample residual slope predictions by drawing from distribution
            r_next=r_prev+y_s*DT_H    # update residual
            T_pred=T_rc1+r_next    # Predict indoor temperature: T_predicted = T_RC + residual
            samples.append(T_pred) # repeat M times to get a distribution of outcomes
    S=np.stack(samples,0)
    return S.mean(axis=0),S.std(axis=0),T_true,T_rc1,ends    # return mean of S MC samples


# 10) Metrics
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))    # magnitude of error between predictions a and ground truth b
def cvrmse(a,b): return 100.0*rmse(a,b)/float(np.mean(b))    # normalised RMSE by mean of observed data (as a %)

