"""
hybrid_model.py
================================
Independent residual-level hybrid model. Patched to add: uncertainty-source decomposition (RC epistemic /
LSTM epistemic / LSTM aleatoric / full), PICP for rc_full and hybrid_full,
ADVI degeneracy flagging, and the extra metadata Phase 1 needs.

Usage:
    python hybrid_model.py --igl {} --input IGL{}.csv --metadata Metadata.xlsx
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import RC_model as rcm

DTYPE = rcm.DTYPE
DEVICE = rcm.DEVICE

# =============================================================
# Hybrid-specific settings
# =============================================================
CONTEXT_HOURS = 24.0
EVAL_HOURS = 96.0

LSTM_HIDDEN = 64
DROPOUT_P = 0.25
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.15

S_DEFAULT = rcm.N_POSTERIOR_DRAWS
HORIZON_CHECKPOINTS_H = rcm.HORIZON_CHECKPOINTS_H

SEED = 42
MIN_EXTRA_TRAIN_DAYS = 3.0

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_FEATURE_COLS = ["To", "Irr", "Qint", "Qah", "Ria", "Tin_rc", "dTin_rc",
                      "h_sin", "h_cos", "dow_sin", "dow_cos"]
SOLAR_FEATURE_COLS = ["el_sin", "el_cos", "az_sin", "az_cos"]
IDX_TIN_RC = 5
IDX_DTIN_RC = 6


# =============================================================
# Time-of-day / day-of-week / solar-angle features (unchanged)
# =============================================================
def build_time_solar_features(df):
    idx = df.index
    hours = idx.hour + idx.minute / 60.0 + idx.second / 3600.0
    h_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    h_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
    dow = idx.weekday.values.astype(np.float32)
    dow_sin = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

    el_candidates = ["solar_elevation_deg", "solar_elevation", "elevation_deg"]
    az_candidates = ["solar_azimuth_deg", "solar_azimuth", "azimuth_deg"]
    el_col = next((c for c in el_candidates if c in df.columns), None)
    az_col = next((c for c in az_candidates if c in df.columns), None)

    if el_col is None or az_col is None:
        print("[warn] Solar elevation/azimuth columns not found -- proceeding without sun-angle features.")
        return dict(h_sin=h_sin, h_cos=h_cos, dow_sin=dow_sin, dow_cos=dow_cos), False

    el = pd.to_numeric(df[el_col], errors="coerce").to_numpy(np.float32)
    az = pd.to_numeric(df[az_col], errors="coerce").to_numpy(np.float32)
    el_rad, az_rad = np.deg2rad(el), np.deg2rad(az)
    el_sin, el_cos = np.sin(el_rad).astype(np.float32), np.cos(el_rad).astype(np.float32)
    az_sin, az_cos = np.sin(az_rad).astype(np.float32), np.cos(az_rad).astype(np.float32)

    feats = dict(h_sin=h_sin, h_cos=h_cos, dow_sin=dow_sin, dow_cos=dow_cos,
                 el_sin=el_sin, el_cos=el_cos, az_sin=az_sin, az_cos=az_cos)
    return feats, True


def assemble_features(To, Irr, Qint, Qah, Ria, Tin_rc, dTin_rc, tf, solar_available):
    cols = [To, Irr, Qint, Qah, Ria, Tin_rc, dTin_rc,
            tf["h_sin"], tf["h_cos"], tf["dow_sin"], tf["dow_cos"]]
    if solar_available:
        cols += [tf["el_sin"], tf["el_cos"], tf["az_sin"], tf["az_cos"]]
    return np.column_stack(cols).astype(np.float32)


# =============================================================
# Mode A -- deterministic continuous RC rollout (posterior mean)
# =============================================================
def params_to_tensor(P_post):
    return {k: torch.tensor(float(v), dtype=DTYPE, device=DEVICE) for k, v in P_post.items()}


def mode_a_rollout(rc, P_post, Tin0_val, To, Irr, Qint, Qah, Ria):
    params_t = params_to_tensor(P_post)
    Tin0 = torch.tensor(float(Tin0_val), dtype=DTYPE, device=DEVICE)
    To_t = torch.tensor(To, dtype=DTYPE, device=DEVICE)
    Irr_t = torch.tensor(Irr, dtype=DTYPE, device=DEVICE)
    Qint_t = torch.tensor(Qint, dtype=DTYPE, device=DEVICE)
    Qah_t = torch.tensor(Qah, dtype=DTYPE, device=DEVICE)
    Ria_t = torch.tensor(Ria, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        out = rc.rollout(params_t, Tin0, To_t, Irr_t, Qint_t, Qah_t, Ria_t)
    return out.cpu().numpy()


# =============================================================
# Windowed dataset -- target is residual LEVEL (degC)
# =============================================================
def build_window_ends(Xn, y, seq_len):
    n = len(y)
    ends = []
    for e in range(seq_len - 1, n):
        s = e - (seq_len - 1)
        if np.isfinite(Xn[s:e + 1]).all() and np.isfinite(y[e]):
            ends.append(e)
    return np.asarray(ends, dtype=int)


class ResidualWindowDataset(Dataset):
    def __init__(self, Xn, y, ends, seq_len):
        self.Xn, self.y, self.ends, self.seq_len = Xn, y.astype(np.float32), ends, seq_len

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        e = self.ends[i]
        s = e - (self.seq_len - 1)
        Xw = self.Xn[s:e + 1]
        return torch.tensor(Xw, dtype=DTYPE), torch.tensor([self.y[e]], dtype=DTYPE)


# =============================================================
# Model
# =============================================================
class HybridLSTM(nn.Module):
    def __init__(self, in_dim, hidden=LSTM_HIDDEN, dropout=DROPOUT_P):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mean_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)
        self.logvar_bias = nn.Parameter(torch.tensor(-1.0, dtype=DTYPE))

    def forward(self, x):
        h, _ = self.lstm(x)
        h_last = self.drop(h[:, -1, :])
        mu = self.mean_head(h_last)
        logvar = torch.clamp(self.logvar_head(h_last) + self.logvar_bias, -8.0, 6.0)
        return mu, logvar


def gaussian_nll(y, mu, logvar):
    return 0.5 * torch.mean((y - mu) ** 2 * torch.exp(-logvar) + logvar)


def train_hybrid_lstm(model, train_ds, val_ds, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                       batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                         patience=3, min_lr=1e-5)

    best_val = float("inf")
    bad = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            mu, logvar = model(xb)
            loss = gaussian_nll(yb, mu, logvar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n_tr += xb.size(0)

        model.eval()
        va_loss, n_va = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mu, logvar = model(xb)
                loss = gaussian_nll(yb, mu, logvar)
                va_loss += loss.item() * xb.size(0)
                n_va += xb.size(0)

        tr_loss /= max(1, n_tr)
        va_loss /= max(1, n_va)
        sched.step(va_loss)

        if va_loss + 1e-6 < best_val:
            best_val, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep == 1 or ep % 5 == 0:
            print(f"[LSTM] epoch {ep:3d} | train NLL={tr_loss:.4f} | val NLL={va_loss:.4f} | "
                  f"lr={opt.param_groups[0]['lr']:.2e} | bad={bad}")
        if bad >= patience:
            print(f"[LSTM] early stopping at epoch {ep} (no val improvement for {patience} epochs).")
            break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"[LSTM] training complete in {time.time() - t0:.1f}s | best val NLL={best_val:.4f}")
    return model


# =============================================================
# Independent per-step rollout -- NOW with dropout / residual_noise toggles
# so the same function serves the full stochastic forecast AND the three
# isolated-source decomposition rollouts (mirrors recursive_hybrid_rollout
# in hybrid_model.py, adapted for this architecture's no-accumulation rule).
# =============================================================
def independent_hybrid_rollout(model, Tin_rc_eval_raw, Tin_rc_eval_norm, dTin_rc_eval_norm,
                                shared_eval_norm, lookback_n, seq_len, S, seed=SEED,
                                dropout=True, residual_noise=True):
    torch.manual_seed(seed)
    T_eval, Fdim = shared_eval_norm.shape

    eval_features_norm = shared_eval_norm.unsqueeze(1).expand(-1, S, -1).clone()
    eval_features_norm[:, :, IDX_TIN_RC] = Tin_rc_eval_norm
    eval_features_norm[:, :, IDX_DTIN_RC] = dTin_rc_eval_norm

    buffer = torch.zeros(S, seq_len, Fdim, dtype=DTYPE, device=DEVICE)
    buffer[:, :seq_len - 1, :] = lookback_n.unsqueeze(0).expand(S, -1, -1)

    Tin_pred = torch.empty(T_eval, S, dtype=DTYPE, device=DEVICE)

    model.train() if dropout else model.eval()
    with torch.no_grad():
        for t in range(T_eval):
            buffer = torch.cat([buffer[:, 1:, :], eval_features_norm[t].unsqueeze(1)], dim=1)
            mu, logvar = model(buffer)
            mu, logvar = mu.squeeze(-1), logvar.squeeze(-1)
            if residual_noise:
                eps = torch.randn(S, dtype=DTYPE, device=DEVICE)
                residual_sample = mu + torch.exp(0.5 * logvar) * eps
            else:
                residual_sample = mu
            Tin_pred[t] = Tin_rc_eval_raw[t] + residual_sample  # independent: no state carried forward

    model.train()  # restore default state for any subsequent call
    return Tin_pred.cpu().numpy()


def compute_horizon_table(mean_pred, true_vals, dt_h, horizons=HORIZON_CHECKPOINTS_H):
    T = len(true_vals)
    horizon_h = np.arange(T) * dt_h
    overall = dict(rmse=rcm.rmse(mean_pred, true_vals), cvrmse=rcm.cvrmse(mean_pred, true_vals), n_steps=T)
    rows = []
    for h in horizons:
        idx = np.searchsorted(horizon_h, h, side="right")
        if idx < 5:
            continue
        rows.append(dict(horizon_h=h,
                          rmse=rcm.rmse(mean_pred[:idx], true_vals[:idx]),
                          cvrmse=rcm.cvrmse(mean_pred[:idx], true_vals[:idx]),
                          n_steps=idx))
    return overall, pd.DataFrame(rows)


# =============================================================
# Main
# =============================================================
def run_hybrid(igl, input_file, metadata_path, S=S_DEFAULT):
    print(f"\n{'=' * 70}\nHybrid RC + LSTM (independent residual-level variant) -- IGL{igl}\n{'=' * 70}")

    home_priors = rcm.load_home_priors(metadata_path, igl)
    Az = home_priors["glazed_area"]
    df, req, dt_s = rcm.load_and_validate(input_file)
    dt_h = dt_s / 3600.0

    time_solar_features, solar_available = build_time_solar_features(df)
    feature_cols = BASE_FEATURE_COLS + (SOLAR_FEATURE_COLS if solar_available else [])

    Tin = df[rcm.COLS["Tin"]].to_numpy(np.float32)
    To = df[rcm.COLS["To"]].to_numpy(np.float32)
    Irr = df[rcm.COLS["I"]].to_numpy(np.float32)
    Qint = df[rcm.COLS["Qint"]].to_numpy(np.float32)
    Qah = df[rcm.COLS["Qah"]].to_numpy(np.float32)
    Ria = df[rcm.COLS["Ria"]].to_numpy(np.float32)

    finite = df[req].notna().all(axis=1).to_numpy()
    main_start_i, main_end_i = rcm.longest_finite_run(finite, start_after=0)
    if main_start_i is None:
        raise RuntimeError("[data] No finite run found in this home's data at all.")
    main_len_steps = main_end_i - main_start_i

    steps_week = int(pd.Timedelta(days=rcm.ADVI_DAYS).total_seconds() // dt_s)
    eval_steps = int(round(pd.Timedelta(hours=EVAL_HOURS).total_seconds() / dt_s))

    if main_len_steps <= steps_week + eval_steps:
        raise RuntimeError(
            f"[data] Main usable block only has {main_len_steps} steps "
            f"({main_len_steps * dt_s / 3600 / 24:.2f} days); need at least "
            f"{rcm.ADVI_DAYS} days ADVI + {EVAL_HOURS:.0f}h eval, with nothing left for training."
        )

    advi_start_i = main_start_i
    advi_end_i = advi_start_i + steps_week
    eval_end_i = main_end_i
    eval_start_i = eval_end_i - eval_steps
    train_start_i = advi_start_i
    train_end_i = eval_start_i

    extra_train_days = (train_end_i - advi_end_i) * dt_s / 3600 / 24
    if extra_train_days < MIN_EXTRA_TRAIN_DAYS:
        print(f"[warn] Only {extra_train_days:.1f} days of training data beyond the ADVI week itself.")

    print(f"[windows] ADVI: {df.index[advi_start_i]} -> {df.index[advi_end_i-1]} | "
          f"train: {df.index[train_start_i]} -> {df.index[train_end_i-1]} "
          f"({(train_end_i-train_start_i)*dt_s/3600/24:.2f} days) | "
          f"eval: {df.index[eval_start_i]} -> {df.index[eval_end_i-1]} "
          f"({(eval_end_i-eval_start_i)*dt_s/3600:.1f}h)")

    # ---- Stage 0: RC ADVI calibration ----
    rc = rcm.RCBackwardEuler(dt_s, rcm.G_TRANSMITTANCE, Az)
    advi_priors = rcm.build_advi_priors(home_priors)
    P_post, q_post = rcm.advi_rc_week1(
        rc, advi_priors,
        Tin[advi_start_i:advi_end_i], To[advi_start_i:advi_end_i], Irr[advi_start_i:advi_end_i],
        Qint[advi_start_i:advi_end_i], Qah[advi_start_i:advi_end_i], Ria[advi_start_i:advi_end_i],
    )
    print("\nPosterior mean RC parameters:")
    for k, v in P_post.items():
        print(f"  {k:10s} = {v:.6g}")

    prior_means_check = {"R_ie": home_priors["r_ie"], "R_ea": home_priors["r_ea"],
                          "C_in": home_priors["c_in"], "C_en": 55_500_000}
    flagged_params = rcm.check_advi_degenerate(P_post, prior_means_check, igl=igl)

    # ---- Stage 1: Mode A -- deterministic CONTINUOUS rollout over training block ----
    Tin_block = Tin[train_start_i:train_end_i]
    Tin_rc_block = mode_a_rollout(
        rc, P_post, Tin_block[0],
        To[train_start_i:train_end_i], Irr[train_start_i:train_end_i],
        Qint[train_start_i:train_end_i], Qah[train_start_i:train_end_i], Ria[train_start_i:train_end_i],
    )

    dTin_rc = np.diff(Tin_rc_block) / dt_h
    y_train = (Tin_block[1:] - Tin_rc_block[1:]).astype(np.float32)  # residual level, degC

    tf_train = {k: v[train_start_i + 1:train_end_i] for k, v in time_solar_features.items()}
    X_train = assemble_features(
        To[train_start_i + 1:train_end_i], Irr[train_start_i + 1:train_end_i],
        Qint[train_start_i + 1:train_end_i], Qah[train_start_i + 1:train_end_i],
        Ria[train_start_i + 1:train_end_i], Tin_rc_block[1:], dTin_rc,
        tf_train, solar_available,
    )

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    Xn_train = (X_train - mu) / sd

    seq_len = max(2, int(round(CONTEXT_HOURS * 3600.0 / dt_s)))
    ends = build_window_ends(Xn_train, y_train, seq_len)
    if len(ends) < 20:
        raise RuntimeError(f"[data] Only {len(ends)} usable training windows for IGL{igl}.")

    n_val = max(1, int(round(VAL_FRACTION * len(ends))))
    train_ends, val_ends = ends[:-n_val], ends[-n_val:]
    print(f"[LSTM data] {len(train_ends)} train windows | {len(val_ends)} val windows "
          f"(seq_len={seq_len} steps = {CONTEXT_HOURS:.0f}h context)")
    print(f"[target] residual LEVEL stats (degC): mean={y_train.mean():.4f} std={y_train.std():.4f} "
          f"min={y_train.min():.4f} max={y_train.max():.4f}")

    train_ds = ResidualWindowDataset(Xn_train, y_train, train_ends, seq_len)
    val_ds = ResidualWindowDataset(Xn_train, y_train, val_ends, seq_len)

    # ---- Stage 3: train the LSTM ----
    model = HybridLSTM(in_dim=len(feature_cols)).to(DEVICE)
    model = train_hybrid_lstm(model, train_ds, val_ds)

    # ---- Stage 4: Mode B -- batched S-draw RC rollout over the eval window ----
    torch.manual_seed(SEED)
    draws, _sigmas = rcm.sample_posterior_draws(q_post, S)
    Tin0_eval = torch.full((S,), float(Tin[eval_start_i]), dtype=DTYPE, device=DEVICE)
    To_eval_t = torch.tensor(To[eval_start_i:eval_end_i], dtype=DTYPE, device=DEVICE)
    Irr_eval_t = torch.tensor(Irr[eval_start_i:eval_end_i], dtype=DTYPE, device=DEVICE)
    Qint_eval_t = torch.tensor(Qint[eval_start_i:eval_end_i], dtype=DTYPE, device=DEVICE)
    Qah_eval_t = torch.tensor(Qah[eval_start_i:eval_end_i], dtype=DTYPE, device=DEVICE)
    Ria_eval_t = torch.tensor(Ria[eval_start_i:eval_end_i], dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        Tin_rc_eval = rc.rollout(draws, Tin0_eval, To_eval_t, Irr_eval_t, Qint_eval_t, Qah_eval_t, Ria_eval_t)

    prev_row = torch.full((1, S), float(Tin_rc_block[-1]), dtype=DTYPE, device=DEVICE)
    Tin_rc_eval_ext = torch.cat([prev_row, Tin_rc_eval], dim=0)
    dTin_rc_eval = (Tin_rc_eval_ext[1:] - Tin_rc_eval_ext[:-1]) / dt_h

    mu_t = torch.tensor(mu, dtype=DTYPE, device=DEVICE)
    sd_t = torch.tensor(sd, dtype=DTYPE, device=DEVICE)
    Tin_rc_eval_n = (Tin_rc_eval - mu_t[IDX_TIN_RC]) / sd_t[IDX_TIN_RC]
    dTin_rc_eval_n = (dTin_rc_eval - mu_t[IDX_DTIN_RC]) / sd_t[IDX_DTIN_RC]

    T_eval = eval_end_i - eval_start_i
    zeros_eval = np.zeros(T_eval, dtype=np.float32)
    tf_eval = {k: v[eval_start_i:eval_end_i] for k, v in time_solar_features.items()}
    shared_raw = assemble_features(
        To[eval_start_i:eval_end_i], Irr[eval_start_i:eval_end_i],
        Qint[eval_start_i:eval_end_i], Qah[eval_start_i:eval_end_i], Ria[eval_start_i:eval_end_i],
        zeros_eval, zeros_eval, tf_eval, solar_available,
    )
    shared_norm = (torch.tensor(shared_raw, dtype=DTYPE, device=DEVICE) - mu_t) / sd_t

    lookback_raw = X_train[-(seq_len - 1):]
    lookback_n = torch.tensor((lookback_raw - mu) / sd, dtype=DTYPE, device=DEVICE)

    # ---- Stage 5: independent per-step rollout (full stochastic) ----
    t0 = time.time()
    Tin_pred = independent_hybrid_rollout(
        model, Tin_rc_eval, Tin_rc_eval_n, dTin_rc_eval_n,
        shared_norm, lookback_n, seq_len, S,
        dropout=True, residual_noise=True,
    )
    print(f"[eval] independent {T_eval}-step x S={S} rollout complete in {time.time()-t0:.1f}s")

    mean_pred = Tin_pred.mean(axis=1)
    std_pred = Tin_pred.std(axis=1)
    Tin_true_eval = Tin[eval_start_i:eval_end_i]

    # ---- Sanity check: naive constant-offset baseline ----
    rc_mean_draw = Tin_rc_eval.mean(dim=1).cpu().numpy()
    constant_offset = float(y_train.mean())
    naive_pred = rc_mean_draw + constant_offset
    naive_rmse = rcm.rmse(naive_pred, Tin_true_eval)
    naive_cvrmse = rcm.cvrmse(naive_pred, Tin_true_eval)
    print(f"\n[sanity] Naive constant-offset baseline (RC + {constant_offset:+.3f} degC): "
          f"RMSE={naive_rmse:.3f} degC | CVRMSE={naive_cvrmse:.2f}%")

    # ---- Stage 5b: uncertainty decomposition (3 extra cheap inference-only rollouts) ----
    Tin_rc_eval_mean_1d = mode_a_rollout(
        rc, P_post, Tin[eval_start_i],
        To[eval_start_i:eval_end_i], Irr[eval_start_i:eval_end_i],
        Qint[eval_start_i:eval_end_i], Qah[eval_start_i:eval_end_i], Ria[eval_start_i:eval_end_i],
    )
    Tin_rc_eval_mean_raw = torch.tensor(Tin_rc_eval_mean_1d, dtype=DTYPE, device=DEVICE).unsqueeze(1).expand(-1, S).clone()
    prev_val = torch.full((1, S), float(Tin_rc_block[-1]), dtype=DTYPE, device=DEVICE)
    Tin_rc_eval_mean_ext = torch.cat([prev_val, Tin_rc_eval_mean_raw], dim=0)
    dTin_rc_eval_mean = (Tin_rc_eval_mean_ext[1:] - Tin_rc_eval_mean_ext[:-1]) / dt_h
    Tin_rc_eval_mean_n = (Tin_rc_eval_mean_raw - mu_t[IDX_TIN_RC]) / sd_t[IDX_TIN_RC]
    dTin_rc_eval_mean_n = (dTin_rc_eval_mean - mu_t[IDX_DTIN_RC]) / sd_t[IDX_DTIN_RC]

    decomp = {"hybrid_full": Tin_pred}

    decomp["hybrid_rc_only"] = independent_hybrid_rollout(
        model, Tin_rc_eval, Tin_rc_eval_n, dTin_rc_eval_n, shared_norm, lookback_n, seq_len, S,
        dropout=False, residual_noise=False)

    decomp["hybrid_lstm_epistemic_only"] = independent_hybrid_rollout(
        model, Tin_rc_eval_mean_raw, Tin_rc_eval_mean_n, dTin_rc_eval_mean_n, shared_norm, lookback_n, seq_len, S,
        dropout=True, residual_noise=False)

    decomp["hybrid_lstm_aleatoric_only"] = independent_hybrid_rollout(
        model, Tin_rc_eval_mean_raw, Tin_rc_eval_mean_n, dTin_rc_eval_mean_n, shared_norm, lookback_n, seq_len, S,
        dropout=False, residual_noise=True)

    hybrid_overall, hybrid_horizon = compute_horizon_table(mean_pred, Tin_true_eval, dt_h)

    # RC-only baseline over the identical eval window
    rc_mean, rc_std, rc_std_epistemic, rc_overall, rc_horizon, horizon_h_arr, rc_picp_overall, rc_picp_horizon = \
        rcm.evaluate_free_run(
            rc, q_post, Tin_true_eval,
            To[eval_start_i:eval_end_i], Irr[eval_start_i:eval_end_i],
            Qint[eval_start_i:eval_end_i], Qah[eval_start_i:eval_end_i], Ria[eval_start_i:eval_end_i],
            dt_s, S=S, return_draws=True,
        )

    hybrid_picp_overall, hybrid_picp_horizon = rcm.compute_picp(decomp["hybrid_full"], Tin_true_eval, dt_h)

    print(f"\n=== Overall eval-window comparison (IGL{igl}) ===")
    print(f"RC-only  : RMSE={rc_overall['rmse']:.3f} degC | CVRMSE={rc_overall['cvrmse']:.2f}%")
    print(f"Hybrid   : RMSE={hybrid_overall['rmse']:.3f} degC | CVRMSE={hybrid_overall['cvrmse']:.2f}%")
    print(f"Naive constant-offset: RMSE={naive_rmse:.3f} degC | CVRMSE={naive_cvrmse:.2f}% "
          f"(sanity check -- LSTM should beat this)")
    for lvl in rcm.PICP_LEVELS:
        print(f"  PICP @ {int(lvl*100)}%: RC={rc_picp_overall[lvl]['picp']*100:.1f}% | "
              f"Hybrid={hybrid_picp_overall[lvl]['picp']*100:.1f}%")

    merged = pd.merge(
        rc_horizon.rename(columns={"rmse": "rc_rmse", "cvrmse": "rc_cvrmse", "n_steps": "rc_n_steps"}),
        hybrid_horizon.rename(columns={"rmse": "hyb_rmse", "cvrmse": "hyb_cvrmse", "n_steps": "hyb_n_steps"}),
        on="horizon_h",
    )
    merged["rmse_improvement_%"] = 100.0 * (merged["rc_rmse"] - merged["hyb_rmse"]) / merged["rc_rmse"]
    print("\nRMSE / CVRMSE by forecast horizon, RC-only vs Hybrid:")
    print(merged.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return dict(
        igl=igl, architecture="residual_level",
        P_post=P_post, q_post=q_post,
        home_priors=home_priors, dt_s=dt_s, Az=Az,
        advi_window=(df.index[advi_start_i], df.index[advi_end_i - 1]),
        train_window=(df.index[train_start_i], df.index[train_end_i - 1]),
        eval_window=(df.index[eval_start_i], df.index[eval_end_i - 1]),
        flagged_params=flagged_params,
        rc_overall=rc_overall, rc_horizon=rc_horizon,
        rc_mean=rc_mean, rc_std=rc_std, rc_std_epistemic=rc_std_epistemic,
        rc_picp_overall=rc_picp_overall, rc_picp_horizon=rc_picp_horizon,
        hybrid_overall=hybrid_overall, hybrid_horizon=hybrid_horizon,
        mean_pred=mean_pred, std_pred=std_pred, Tin_true_eval=Tin_true_eval,
        horizon_h_arr=horizon_h_arr,
        hybrid_picp_overall=hybrid_picp_overall, hybrid_picp_horizon=hybrid_picp_horizon,
        decomp=decomp,
        naive_offset_rmse=naive_rmse, naive_offset_cvrmse=naive_cvrmse,
        model=model,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--igl", type=int, required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--metadata", default="Metadata.xlsx")
    ap.add_argument("--posterior_draws", type=int, default=S_DEFAULT)
    args = ap.parse_args()

    run_hybrid(args.igl, args.input, args.metadata, S=args.posterior_draws)
