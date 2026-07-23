"""
3R2C_model.py
===========
3R2C RC-network model: ADVI calibration + free-run evaluation, for a single
IGL home. Home-specific physical parameters (window area, R_ie/R_ea/C_in
priors) are read automatically from the metadata file -- no manual editing
needed per home.

Usage:
    python RC_model.py --igl 651 --input IGL651_processed__imputed.csv --metadata Metadata.xlsx
"""

import argparse
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(DTYPE)

# =============================================================
# Fixed run settings
# =============================================================
COLS = dict(
    Tin="indoor_temp",
    To="outdoor_temp",
    I="POA_total",
    Qint="internal_gain",
    Qah="heat_flux",
    Ria="R_ia",
)

G_TRANSMITTANCE = 0.76

ADVI_DAYS = 7
ADVI_DAYS_TRANSFER = 2      # used only by the Phase-2 transfer-learning script
ADVI_ITERS = 3000
ADVI_LR = 1e-2
ADVI_SEED = 42
KL_WARMUP = 800
SEED = 42

N_POSTERIOR_DRAWS = 300
HORIZON_CHECKPOINTS_H = [1, 6, 12, 24, 48, 72, 96]
MIN_EVAL_STEPS = 48
PICP_LEVELS = (0.5, 0.8, 0.9, 0.95)

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================
# Metadata -> home-specific priors
# =============================================================
def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load and normalise the metadata file once. Reused for both per-home
    prior lookup and for building the IGL worklist in Phase 1."""
    path = str(metadata_path)
    meta = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    meta.columns = [_norm_col(c) for c in meta.columns]
    if "igl" not in meta.columns:
        raise KeyError("Metadata file must have an 'IGL' column.")
    meta["igl"] = meta["igl"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return meta


def load_home_priors(metadata_path: str, igl: int) -> dict:
    meta = load_metadata(metadata_path).set_index("igl")
    if igl not in meta.index:
        raise KeyError(f"IGL{igl} not found in metadata file.")
    row = meta.loc[igl]

    defaults = dict(glazed_area=20.0, r_ie=0.0094595, r_ea=0.0094595, c_in=317_137.8)
    out = {}
    for key, default in defaults.items():
        v = row.get(key, np.nan)
        if pd.isna(v):
            print(f"[warn] IGL{igl}: metadata missing '{key}', using default {default}")
            out[key] = default
        else:
            out[key] = float(v)
    return out


# =============================================================
# Timestamp parsing / loading
# =============================================================
def parse_timestamp_col(series):
    s = series.astype(str).str.strip()
    ts_iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ts_ddmm1 = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    ts_ddmm2 = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    ts_any = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return ts_iso.fillna(ts_ddmm1).fillna(ts_ddmm2).fillna(ts_any)


def load_and_validate(input_file):
    df_raw = pd.read_csv(input_file)
    if "timestamp" not in df_raw.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")

    ts = parse_timestamp_col(df_raw["timestamp"])
    if ts.isna().any():
        nbad = int(ts.isna().sum())
        bad = df_raw.loc[ts.isna(), "timestamp"].astype(str).str.strip().unique()[:10]
        raise ValueError(f"[parse] Failed to parse {nbad} timestamps. Examples: {bad}")

    df = df_raw.copy()
    df["timestamp"] = ts
    df = df.sort_values("timestamp").set_index("timestamp")

    req = [COLS["Tin"], COLS["To"], COLS["I"], COLS["Qint"], COLS["Qah"], COLS["Ria"]]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"Required column not found: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dsec = df.index.to_series().diff().dt.total_seconds().to_numpy()
    dsec_pos = dsec[np.isfinite(dsec) & (dsec > 0)]
    if dsec_pos.size == 0:
        raise RuntimeError("[validate] Need >=2 increasing timestamps to infer cadence.")
    dt_s = float(np.median(dsec_pos))

    print(f"[timeline] {df.index.min()} -> {df.index.max()}  ({len(df)} rows)")
    print(f"[cadence] median dt ~= {dt_s/3600:.3f} h ({dt_s:.0f} s)")
    return df, req, dt_s


def find_finite_run(mask: np.ndarray, min_len: int = 1, start_after: int = 0):
    n = len(mask)
    i = start_after
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_len:
            return i, j
        i = j
    return None, None


def longest_finite_run(mask: np.ndarray, start_after: int = 0):
    n = len(mask)
    best = (None, None, 0)
    i = start_after
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) > best[2]:
            best = (i, j, j - i)
        i = j
    return best[0], best[1]


# =============================================================
# RC model -- analytic 2x2 Backward Euler (hoisted loop-invariants)
# =============================================================
class RCBackwardEuler:
    def __init__(self, dt_s, g=G_TRANSMITTANCE, Az=41.3):
        self.dt = float(dt_s)
        self.g = float(g)
        self.Az = float(Az)

    def rollout(self, params, Tin0, To, Irr, Qint, Qah, Ria, tin_clip=(5.0, 45.0)):
        dt = self.dt
        R_ie, R_ea, C_in, C_en = params["R_ie"], params["R_ea"], params["C_in"], params["C_en"]
        a_si, a_se = params["a_sol_in"], params["a_sol_en"]
        a_ii, a_ie = params["a_int_in"], params["a_int_en"]

        Te0 = Tin0.clone() if torch.is_tensor(Tin0) else torch.tensor(Tin0, dtype=DTYPE, device=DEVICE)
        batched = Te0.dim() > 0

        Irr_c = torch.clamp(Irr, min=0.0, max=1200.0)
        Qah_c = torch.clamp(Qah, min=0.0)
        Ria_c = torch.clamp(Ria, min=1e-4)
        Qsol = self.g * self.Az * Irr_c

        if batched:
            To_b, Qsol_b, Qint_b, Qah_c_b, Ria_c_b = (
                x.unsqueeze(-1) for x in (To, Qsol, Qint, Qah_c, Ria_c)
            )
        else:
            To_b, Qsol_b, Qint_b, Qah_c_b, Ria_c_b = To, Qsol, Qint, Qah_c, Ria_c

        A11 = -(1.0 / R_ea + 1.0 / R_ie) / C_en
        A12 = (1.0 / R_ie) / C_en
        A21 = (1.0 / R_ie) / C_in
        m11 = 1.0 - dt * A11
        m12 = -dt * A12
        m21 = -dt * A21
        B11 = 1.0 / (C_en * R_ea)
        B12 = a_se / C_en
        B13 = a_ie / C_en
        B22 = a_si / C_in
        B23 = a_ii / C_in
        B24 = 1.0 / C_in

        forcing1 = dt * (B11 * To_b + B12 * Qsol_b + B13 * Qint_b)
        forcing2 = dt * (To_b / (C_in * Ria_c_b) + B22 * Qsol_b + B23 * Qint_b + B24 * Qah_c_b)

        A22_series = -(1.0 / Ria_c_b + 1.0 / R_ie) / C_in
        m22_series = 1.0 - dt * A22_series
        det_series = m11 * m22_series - m12 * m21

        Te = Te0.clone()
        Tin = Te0.clone()
        T = To.shape[0]
        out = torch.empty((T,) + tuple(Te.shape), dtype=DTYPE, device=DEVICE)

        for k in range(T):
            rhs1 = Te + forcing1[k]
            rhs2 = Tin + forcing2[k]
            m22, det = m22_series[k], det_series[k]

            Te_new = (m22 * rhs1 - m12 * rhs2) / det
            Tin_new = (-m21 * rhs1 + m11 * rhs2) / det

            Te, Tin = Te_new, torch.clamp(Tin_new, min=tin_clip[0], max=tin_clip[1])
            out[k] = Tin

        return out


# =============================================================
# ADVI calibration
# =============================================================
def logit(p):
    return np.log(p / (1 - p))


def build_advi_priors(home_priors):
    return dict(
        R_ie=dict(mean=home_priors["r_ie"], sd_uncon=1.0),
        R_ea=dict(mean=home_priors["r_ea"], sd_uncon=1.0),
        C_in=dict(mean=home_priors["c_in"], sd_uncon=1.0),
        C_en=dict(mean=55_500_000, sd_uncon=1.0),
        a_sol_in=dict(mean=0.5, sd_uncon=0.7),
        a_sol_en=dict(mean=0.5, sd_uncon=0.7),
        a_int_in=dict(mean=0.5, sd_uncon=0.7),
        a_int_en=dict(mean=0.5, sd_uncon=0.7),
        sigma=dict(mean=1.5, sd_uncon=0.6),
    )


def advi_rc_week1(rc, advi_priors, Tin_tr_np, To_tr_np, Irr_tr_np, Qint_tr_np, Qah_tr_np, Ria_tr_np,
                   iters=ADVI_ITERS, lr=ADVI_LR, kl_warmup_iters=KL_WARMUP, seed=ADVI_SEED,
                   sigma_bounds=(1e-3, 50.0), prior_mu_uncon=None, prior_sd_uncon=None):
    """
    advi_priors: home-specific priors dict (from build_advi_priors), used when
                 prior_mu_uncon is None. Pass None for advi_priors when supplying
                 prior_mu_uncon/prior_sd_uncon directly (Phase-2 transfer learning).
    prior_mu_uncon/prior_sd_uncon: optional dicts, already in UNCONSTRAINED space
                 (log for R/C, logit for absorption coefficients, log for sigma),
                 overriding advi_priors entirely when supplied.
    """
    torch.manual_seed(seed)

    Tin_tr = torch.tensor(Tin_tr_np, dtype=DTYPE, device=DEVICE)
    To_tr = torch.tensor(To_tr_np, dtype=DTYPE, device=DEVICE)
    Irr_tr = torch.tensor(Irr_tr_np, dtype=DTYPE, device=DEVICE)
    Qint_tr = torch.tensor(Qint_tr_np, dtype=DTYPE, device=DEVICE)
    Qah_tr = torch.tensor(Qah_tr_np, dtype=DTYPE, device=DEVICE)
    Ria_tr = torch.tensor(Ria_tr_np, dtype=DTYPE, device=DEVICE)
    Tin0 = Tin_tr[0]

    for name, t in dict(Tin=Tin_tr, To=To_tr, Irr=Irr_tr, Qint=Qint_tr, Qah=Qah_tr, Ria=Ria_tr).items():
        if not torch.isfinite(t).all():
            raise ValueError(f"[ADVI] Non-finite values in ADVI window: {name}")

    def make_var_uncon(mu0):
        mu = torch.tensor(mu0, dtype=DTYPE, device=DEVICE, requires_grad=True)
        rho = torch.tensor(-2.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
        return dict(mu=mu, rho=rho)

    if prior_mu_uncon is None:
        prior_mu, prior_sd = {}, {}
        for name in ["R_ie", "R_ea", "C_in", "C_en"]:
            prior_mu[name] = torch.tensor(np.log(advi_priors[name]["mean"]), dtype=DTYPE, device=DEVICE)
            prior_sd[name] = torch.tensor(advi_priors[name]["sd_uncon"], dtype=DTYPE, device=DEVICE)
        for name in ["a_sol_in", "a_sol_en", "a_int_in", "a_int_en"]:
            prior_mu[name] = torch.tensor(logit(advi_priors[name]["mean"]), dtype=DTYPE, device=DEVICE)
            prior_sd[name] = torch.tensor(advi_priors[name]["sd_uncon"], dtype=DTYPE, device=DEVICE)
        prior_mu["log_sigma"] = torch.tensor(np.log(advi_priors["sigma"]["mean"]), dtype=DTYPE, device=DEVICE)
        prior_sd["log_sigma"] = torch.tensor(advi_priors["sigma"]["sd_uncon"], dtype=DTYPE, device=DEVICE)
    else:
        prior_mu = {n: torch.tensor(prior_mu_uncon[n], dtype=DTYPE, device=DEVICE) for n in prior_mu_uncon}
        prior_sd = {n: torch.tensor(prior_sd_uncon[n], dtype=DTYPE, device=DEVICE) for n in prior_sd_uncon}

    q = {name: make_var_uncon(prior_mu[name].item()) for name in prior_mu}
    opt = torch.optim.Adam([p for d in q.values() for p in d.values()], lr=lr)

    def sample_params_and_kl(kl_scale=1.0):
        params, sigma = {}, None
        kl = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
        for name, vr in q.items():
            mu, rho = vr["mu"], vr["rho"]
            std = F.softplus(rho)
            z = mu + std * torch.randn_like(mu)
            pmu, psd = prior_mu[name], prior_sd[name]
            kl += kl_scale * (torch.log(psd / std) + (std ** 2 + (mu - pmu) ** 2) / (2 * psd ** 2) - 0.5)
            if name in ["R_ie", "R_ea", "C_in", "C_en"]:
                params[name] = torch.exp(z)
            elif name in ["a_sol_in", "a_sol_en", "a_int_in", "a_int_en"]:
                params[name] = torch.sigmoid(z)
            elif name == "log_sigma":
                sigma = torch.exp(z).squeeze()
        sigma_eff = torch.clamp(sigma, sigma_bounds[0], sigma_bounds[1])
        return params, sigma_eff, kl

    for it in range(1, iters + 1):
        kl_scale = min(1.0, it / max(1, kl_warmup_iters))
        opt.zero_grad()
        params, sigma_eff, kl = sample_params_and_kl(kl_scale)
        Tin_hat = rc.rollout(params, Tin0, To_tr, Irr_tr, Qint_tr, Qah_tr, Ria_tr)
        if not torch.isfinite(Tin_hat).all():
            raise FloatingPointError("[ADVI] Tin_hat became non-finite; check inputs and R_ia range.")

        ll = -0.5 * torch.sum(((Tin_tr - Tin_hat) / sigma_eff) ** 2 + 2 * torch.log(sigma_eff) + np.log(2 * np.pi))
        elbo = ll - kl
        if not torch.isfinite(elbo):
            raise FloatingPointError("[ADVI] Non-finite ELBO encountered.")

        (-elbo).backward()
        torch.nn.utils.clip_grad_norm_([p for d in q.values() for p in d.values()], max_norm=5.0)
        opt.step()

        if it % 500 == 0:
            print(f"[ADVI] iter {it:4d} | ELBO={elbo.item():.1f} | KL={kl.item():.2f} | "
                  f"sigma={float(sigma_eff.detach()):.3f} | kl_scale={kl_scale:.2f}")

    with torch.no_grad():
        P_post = dict(
            R_ie=float(torch.exp(q["R_ie"]["mu"])), R_ea=float(torch.exp(q["R_ea"]["mu"])),
            C_in=float(torch.exp(q["C_in"]["mu"])), C_en=float(torch.exp(q["C_en"]["mu"])),
            a_sol_in=float(torch.sigmoid(q["a_sol_in"]["mu"])), a_sol_en=float(torch.sigmoid(q["a_sol_en"]["mu"])),
            a_int_in=float(torch.sigmoid(q["a_int_in"]["mu"])), a_int_en=float(torch.sigmoid(q["a_int_en"]["mu"])),
        )
        q_post = {n: dict(mu=q[n]["mu"].detach().clone(), rho=q[n]["rho"].detach().clone()) for n in q}
    return P_post, q_post


def check_advi_degenerate(P_post, prior_means, igl=None):
    """Flags parameters whose posterior mean strayed >100x or <0.01x from its
    prior -- symptomatic of a degenerate/overfit ADVI fit rather than a
    physically meaningful result. Returns a list of flagged parameter names
    (empty list = clean)."""
    flagged_params = []
    for name, prior_mean in prior_means.items():
        ratio = P_post[name] / prior_mean
        if ratio < 0.01 or ratio > 100:
            flagged_params.append(name)
            print(f"  [WARNING] IGL{igl}: {name} posterior ({P_post[name]:.4g}) is {ratio:.3g}x "
                  f"its prior ({prior_mean:.4g}) -- likely degenerate ADVI fit.")
    return flagged_params


# =============================================================
# Vectorized free-run evaluation across posterior draws
# =============================================================
def sample_posterior_draws(q_post, S):
    draws, sigmas = {}, None
    for name, vr in q_post.items():
        mu, rho = vr["mu"], vr["rho"]
        std = F.softplus(rho)
        z = mu + std * torch.randn(S, dtype=DTYPE, device=DEVICE)
        if name in ["R_ie", "R_ea", "C_in", "C_en"]:
            draws[name] = torch.exp(z)
        elif name in ["a_sol_in", "a_sol_en", "a_int_in", "a_int_en"]:
            draws[name] = torch.sigmoid(z)
        elif name == "log_sigma":
            sigmas = torch.exp(z)
    return draws, sigmas


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cvrmse(a, b):
    return 100.0 * rmse(a, b) / float(np.mean(b))


def compute_picp(sims, true_vals, dt_h, levels=PICP_LEVELS, horizons=HORIZON_CHECKPOINTS_H):
    """
    sims: (T, S) array of draws. Empirical-quantile based, no Gaussianity assumed.
    Returns: overall dict {level: {picp, mean_interval_width}}, and a cumulative
             horizon_table DataFrame (same convention as the RMSE horizon table).
    """
    T, S = sims.shape
    true_vals = np.asarray(true_vals)
    overall = {}
    for lvl in levels:
        alpha = 1 - lvl
        lo = np.quantile(sims, alpha / 2, axis=1)
        hi = np.quantile(sims, 1 - alpha / 2, axis=1)
        covered = (true_vals >= lo) & (true_vals <= hi)
        overall[lvl] = dict(picp=float(np.mean(covered)), mean_interval_width=float(np.mean(hi - lo)))

    horizon_h = np.arange(T) * dt_h
    rows = []
    for h in horizons:
        idx = np.searchsorted(horizon_h, h, side="right")
        if idx < 5:
            continue
        for lvl in levels:
            alpha = 1 - lvl
            lo = np.quantile(sims[:idx], alpha / 2, axis=1)
            hi = np.quantile(sims[:idx], 1 - alpha / 2, axis=1)
            covered = (true_vals[:idx] >= lo) & (true_vals[:idx] <= hi)
            rows.append(dict(horizon_h=h, level=lvl, picp=float(np.mean(covered)),
                              mean_interval_width=float(np.mean(hi - lo)), n_steps=idx))
    return overall, pd.DataFrame(rows)


def evaluate_free_run(rc, q_post, Tin_np, To_np, Irr_np, Qint_np, Qah_np, Ria_np, dt_s,
                       S=N_POSTERIOR_DRAWS, return_draws=False):
    draws, sigmas = sample_posterior_draws(q_post, S)
    Tin0 = torch.full((S,), float(Tin_np[0]), dtype=DTYPE, device=DEVICE)
    To_t = torch.tensor(To_np, dtype=DTYPE, device=DEVICE)
    Irr_t = torch.tensor(Irr_np, dtype=DTYPE, device=DEVICE)
    Qint_t = torch.tensor(Qint_np, dtype=DTYPE, device=DEVICE)
    Qah_t = torch.tensor(Qah_np, dtype=DTYPE, device=DEVICE)
    Ria_t = torch.tensor(Ria_np, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        sims_epistemic = rc.rollout(draws, Tin0, To_t, Irr_t, Qint_t, Qah_t, Ria_t)  # (T,S), no measurement noise
        if sigmas is not None:
            sims_full = sims_epistemic + torch.randn_like(sims_epistemic) * sigmas.unsqueeze(0)
        else:
            sims_full = sims_epistemic
        sims_epistemic_np = sims_epistemic.cpu().numpy()
        sims_full_np = sims_full.cpu().numpy()

    mean_pred = sims_full_np.mean(axis=1)
    std_pred = sims_full_np.std(axis=1)
    std_pred_epistemic = sims_epistemic_np.std(axis=1)

    T = len(Tin_np)
    dt_h = dt_s / 3600.0
    horizon_h = np.arange(T) * dt_h

    overall = dict(rmse=rmse(mean_pred, Tin_np), cvrmse=cvrmse(mean_pred, Tin_np), n_steps=T)

    rows = []
    for h in HORIZON_CHECKPOINTS_H:
        idx = np.searchsorted(horizon_h, h, side="right")
        if idx < 5:
            continue
        rows.append(dict(horizon_h=h, rmse=rmse(mean_pred[:idx], Tin_np[:idx]),
                          cvrmse=cvrmse(mean_pred[:idx], Tin_np[:idx]), n_steps=idx))
    horizon_table = pd.DataFrame(rows)

    if return_draws:
        picp_overall, picp_horizon = compute_picp(sims_full_np, Tin_np, dt_h)
        return mean_pred, std_pred, std_pred_epistemic, overall, horizon_table, horizon_h, picp_overall, picp_horizon
    return mean_pred, std_pred, overall, horizon_table


# =============================================================
# Main (single-home CLI, kept for standalone testing on your laptop)
# =============================================================
def run_rc(igl, input_file, metadata_path):
    print(f"\n{'=' * 70}\nRC model -- IGL{igl}\n{'=' * 70}")

    home_priors = load_home_priors(metadata_path, igl)
    Az = home_priors["glazed_area"]
    print(f"[metadata] Az (Glazed_Area) = {Az:.2f} m^2 | "
          f"R_ie prior = {home_priors['r_ie']:.6g} | R_ea prior = {home_priors['r_ea']:.6g} | "
          f"C_in prior = {home_priors['c_in']:.6g}")

    df, req, dt_s = load_and_validate(input_file)

    Tin = df[COLS["Tin"]].to_numpy(np.float32)
    To = df[COLS["To"]].to_numpy(np.float32)
    Irr = df[COLS["I"]].to_numpy(np.float32)
    Qint = df[COLS["Qint"]].to_numpy(np.float32)
    Qah = df[COLS["Qah"]].to_numpy(np.float32)
    Ria = df[COLS["Ria"]].to_numpy(np.float32)

    finite = df[req].notna().all(axis=1).to_numpy()

    main_start_i, main_end_i = longest_finite_run(finite, start_after=0)
    if main_start_i is None:
        raise RuntimeError("[data] No finite run found in this home's data at all.")
    main_len_steps = main_end_i - main_start_i
    print(f"[main usable block] {df.index[main_start_i]} -> {df.index[main_end_i-1]} "
          f"({main_len_steps} steps, {main_len_steps*dt_s/3600/24:.2f} days)")

    steps_week = int(pd.Timedelta(days=ADVI_DAYS).total_seconds() // dt_s)
    if main_len_steps < steps_week + MIN_EVAL_STEPS:
        raise RuntimeError(
            f"[ADVI] Main usable block only has {main_len_steps} steps "
            f"({main_len_steps*dt_s/3600/24:.2f} days); need at least "
            f"{ADVI_DAYS} days for ADVI plus a usable evaluation remainder."
        )

    advi_start_i = main_start_i
    advi_end_i = advi_start_i + steps_week
    print(f"[ADVI window] {df.index[advi_start_i]} -> {df.index[advi_end_i-1]}  ({steps_week} steps)")

    rc = RCBackwardEuler(dt_s, G_TRANSMITTANCE, Az)
    advi_priors = build_advi_priors(home_priors)

    P_post, q_post = advi_rc_week1(
        rc, advi_priors,
        Tin[advi_start_i:advi_end_i], To[advi_start_i:advi_end_i], Irr[advi_start_i:advi_end_i],
        Qint[advi_start_i:advi_end_i], Qah[advi_start_i:advi_end_i], Ria[advi_start_i:advi_end_i],
    )
    print("\nPosterior mean parameters (RC):")
    for k, v in P_post.items():
        print(f"  {k:10s} = {v:.6g}")

    prior_means = {"R_ie": home_priors["r_ie"], "R_ea": home_priors["r_ea"],
                   "C_in": home_priors["c_in"], "C_en": 55_500_000}
    check_advi_degenerate(P_post, prior_means, igl=igl)

    eval_start_i, eval_end_i = advi_end_i, main_end_i
    print(f"\n[eval window] {df.index[eval_start_i]} -> {df.index[eval_end_i-1]}  "
          f"({eval_end_i - eval_start_i} steps, {(eval_end_i - eval_start_i) * dt_s/3600:.1f} h)")

    mean_pred, std_pred, std_pred_epistemic, overall, horizon_table, horizon_h, picp_overall, picp_horizon = \
        evaluate_free_run(
            rc, q_post,
            Tin[eval_start_i:eval_end_i], To[eval_start_i:eval_end_i], Irr[eval_start_i:eval_end_i],
            Qint[eval_start_i:eval_end_i], Qah[eval_start_i:eval_end_i], Ria[eval_start_i:eval_end_i],
            dt_s, return_draws=True,
        )

    print(f"\n=== Free-run evaluation summary (IGL{igl}) ===")
    print(f"Overall: RMSE = {overall['rmse']:.3f} degC | CVRMSE = {overall['cvrmse']:.2f}% "
          f"| n_steps = {overall['n_steps']}")
    for lvl, d in picp_overall.items():
        print(f"  PICP @ {int(lvl*100)}% nominal: {d['picp']*100:.1f}% covered "
              f"(mean width {d['mean_interval_width']:.2f} degC)")
    if not horizon_table.empty:
        print("\nRMSE / CVRMSE by forecast horizon:")
        print(horizon_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return dict(
        igl=igl, P_post=P_post, q_post=q_post,
        advi_window=(df.index[advi_start_i], df.index[advi_end_i - 1]),
        eval_window=(df.index[eval_start_i], df.index[eval_end_i - 1]),
        overall=overall, horizon_table=horizon_table,
        mean_pred=mean_pred, std_pred=std_pred, std_pred_epistemic=std_pred_epistemic,
        picp_overall=picp_overall, picp_horizon=picp_horizon,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--igl", type=int, required=True)
    ap.add_argument("--input", required=True, help="Path to IGL<n>_processed__imputed.csv")
    ap.add_argument("--metadata", default="Metadata.xlsx")
    args = ap.parse_args()

    run_rc(args.igl, args.input, args.metadata)
