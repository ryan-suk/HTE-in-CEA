# =======================
# STEP-01 (revised): HTE for BCSM, OS, and PROGRESSION @ 5/10/15
# Contrasts:
#   A) S vs OBS (landmark t0 = 60 days)   -> files: s_vs_obs_...csv
#   B) S+RT vs S (landmark t0 = 120 days) -> files: srt_vs_s_...csv
# Learners:
#   - Auto: try econml (DRLearner / ForestDR), else T-learner fallback (two GBMs).
# =======================
import os, sys, warnings, traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import DataConversionWarning

# -------- Engine toggles --------
ENGINE = "auto"               # "auto", "dr", "forestdr", "tlearner"
EXCLUDE_UNKNOWN_TIMING = False
TRIM_LOW, TRIM_HIGH    = 0.02, 0.98

warnings.filterwarnings("ignore", category=DataConversionWarning)

# -------- Paths --------
BASE_DIR   = r"C:\Users\ryann\OneDrive\A_2025-26_Manuscripts\DCIS HTE\SEER files"
PATH_PREP  = os.path.join(BASE_DIR, "dcis_prepared.csv")
PATH_PROG  = os.path.join(BASE_DIR, "dcis_progression.csv")   # REQUIRED
OUT_DIR    = BASE_DIR

# -------- Load prepared --------
df = pd.read_csv(PATH_PREP, dtype=str).rename(columns=lambda c: c.strip())
IDCOL = "patient_id" if "patient_id" in df.columns else ("Patient ID" if "Patient ID" in df.columns else None)
if IDCOL is None:
    raise ValueError("Need patient_id or Patient ID in prepared file.")
df[IDCOL] = df[IDCOL].astype(str)

# -------- Load + merge progression (REQUIRED) --------
if not os.path.exists(PATH_PROG):
    raise FileNotFoundError(
        f"Missing progression file: {PATH_PROG}\n"
        "Create dcis_progression.csv with columns: patient_id (or Patient ID), "
        "event_prog (or progression_event/prog_event), and either prog_days_from_dx "
        "or t_prog_from_60d_days / t_prog_from_120d_days."
    )
prog = pd.read_csv(PATH_PROG, dtype=str).rename(columns=lambda c: c.strip())

# Align ID
PROG_ID = IDCOL if IDCOL in prog.columns else ("Patient ID" if "Patient ID" in prog.columns else None)
if PROG_ID is None:
    raise ValueError("dcis_progression.csv must contain patient_id or Patient ID.")
prog[PROG_ID] = prog[PROG_ID].astype(str)
if PROG_ID != IDCOL:
    prog = prog.rename(columns={PROG_ID: IDCOL})

# Event flag unification
_evt_candidates = ["event_prog","event_progression","progression_event","prog_event"]
_evt_col = next((c for c in _evt_candidates if c in prog.columns), None)
if _evt_col is None:
    raise ValueError(f"dcis_progression.csv must include one of: {', '.join(_evt_candidates)}")
prog["event_prog"] = prog[_evt_col].astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"]).astype(int)

# Times present?
t_from_dx_col  = "prog_days_from_dx"        if "prog_days_from_dx" in prog.columns        else None
t_from_60_col  = "t_prog_from_60d_days"     if "t_prog_from_60d_days" in prog.columns     else None
t_from_120_col = "t_prog_from_120d_days"    if "t_prog_from_120d_days" in prog.columns    else None

# Bring survival follow-up time from prepared (for censoring when dx-based time provided)
surv_days_dx = pd.to_numeric(df.get("surv_days_from_dx"), errors="coerce")

# Merge base progression columns into df
merge_cols = [IDCOL, "event_prog"]
if t_from_dx_col:  merge_cols.append(t_from_dx_col)
if t_from_60_col:  merge_cols.append(t_from_60_col)
if t_from_120_col: merge_cols.append(t_from_120_col)
df = df.merge(prog[merge_cols], on=IDCOL, how="left")

# Build landmarked progression times
if t_from_dx_col:
    # Observed time to progression-or-censor from dx: if event time present, min(event, FU); else FU.
    t_prog_dx = pd.to_numeric(df[t_from_dx_col], errors="coerce")
    df["t_prog_or_cens_from_dx"] = np.where(
        (df["event_prog"].eq(1)) & (t_prog_dx.notna()),
        np.minimum(t_prog_dx, surv_days_dx),
        surv_days_dx
    )
    df["t_prog_from_60d_days"]  = df["t_prog_or_cens_from_dx"] - 60.0
    df["t_prog_from_120d_days"] = df["t_prog_or_cens_from_dx"] - 120.0
else:
    # Rely on landmark columns provided
    if (t_from_60_col is None) and (t_from_120_col is None):
        raise ValueError("Provide either prog_days_from_dx OR t_prog_from_60d_days / t_prog_from_120d_days in progression file.")
    if t_from_60_col:
        df["t_prog_from_60d_days"]  = pd.to_numeric(df[t_from_60_col], errors="coerce")
    if t_from_120_col:
        df["t_prog_from_120d_days"] = pd.to_numeric(df[t_from_120_col], errors="coerce")

# -------- Utilities --------
def to_num(s):  return pd.to_numeric(s, errors="coerce")
def to_bool(s): return s.astype(str).str.strip().str.lower().isin(["true","1","t","y","yes"])

def ipcw_km(time_days, event_indicator, horizon_days, group=None, eps=1e-6):
    """IPCW at fixed horizon H for a 0/1 event."""
    t = np.asarray(to_num(pd.Series(time_days)), dtype=float)
    y = np.asarray(pd.to_numeric(pd.Series(event_indicator), errors="coerce").fillna(0).values, dtype=int)
    H = float(horizon_days)
    valid = np.isfinite(t) & (t >= 0)
    t = t[valid]; y = y[valid]
    tmin = np.minimum(t, H)
    cens = ((t < H) & (y == 0)).astype(int)

    def km_core(tmin_vec, cens_vec):
        n = len(tmin_vec)
        order = np.argsort(tmin_vec, kind="mergesort")
        t_sorted = tmin_vec[order]; e_sorted = cens_vec[order]
        uniq, idx_start = np.unique(t_sorted, return_index=True)
        S = 1.0; G = np.empty(n, float); n_total = n
        for k, start in enumerate(idx_start):
            end = idx_start[k+1] if k+1 < len(idx_start) else n
            n_risk = max(n_total - start, 1); d = int(e_sorted[start:end].sum())
            S *= (1.0 - d / n_risk)
            G[order[start:end]] = max(S, eps)
        return 1.0 / np.clip(G, eps, 1.0)

    if group is None:
        return km_core(tmin, cens), valid
    g = np.asarray(group)[valid]
    W = np.empty_like(tmin, float)
    for gv in np.unique(g):
        idx = (g == gv)
        W[idx] = km_core(tmin[idx], cens[idx])
    return W, valid

# --- NEW: enforce numeric dtypes for design & cate outputs ---
def coerce_design_dtypes(D, idcol):
    """Make inputs_labels_* strictly numeric (except ID)."""
    D = D.copy()
    if "row_id" in D.columns:
        D["row_id"] = pd.to_numeric(D["row_id"], errors="coerce").fillna(0).astype("int64")
    if idcol in D.columns:
        D[idcol] = D[idcol].astype(str)
    for c in ("T", "Y"):
        if c in D.columns:
            D[c] = pd.to_numeric(D[c], errors="coerce").fillna(0).astype("int8")
    if "W" in D.columns:
        D["W"] = pd.to_numeric(D["W"], errors="coerce").fillna(1.0).astype("float32")
    meta = {"row_id", idcol, "T", "Y", "W"}
    feat_cols = [c for c in D.columns if c not in meta]
    for c in feat_cols:
        D[c] = pd.to_numeric(D[c], errors="coerce").fillna(0.0).astype("float32")
    return D

def coerce_cate_dtypes(C, idcol, tau_col):
    """Make cate_* strictly numeric for row_id/tau."""
    C = C.copy()
    if "row_id" in C.columns:
        C["row_id"] = pd.to_numeric(C["row_id"], errors="coerce").fillna(0).astype("int64")
    if idcol in C.columns:
        C[idcol] = C[idcol].astype(str)
    if tau_col in C.columns:
        C[tau_col] = pd.to_numeric(C[tau_col], errors="coerce").astype("float32")
    return C

# -------- Features & labels from prepared --------
any_surgery    = to_bool(df.get("any_surgery", pd.Series(False, index=df.index)))
any_radiation  = to_bool(df.get("any_radiation", pd.Series(False, index=df.index)))
treated_by_t0  = to_bool(df.get("treated_by_t0", pd.Series(False, index=df.index)))
unknown_timing = to_bool(df.get("unknown_timing", pd.Series(False, index=df.index)))

event_bcsm     = to_bool(df.get("event_bcsm", pd.Series(False, index=df.index)))
event_any      = to_bool(df.get("event_any",  pd.Series(False, index=df.index)))
event_prog     = df.get("event_prog", pd.Series(0, index=df.index)).fillna(0).astype(int)

time_A_bcsm_os = to_num(df.get("t_event_from_t0_days"))      # legacy: 60d anchor for BCSM/OS
surv_days_dx   = to_num(df.get("surv_days_from_dx"))         # from dx
time_B_all     = surv_days_dx - 120.0                        # legacy: 120d anchor for BCSM/OS

# NEW: progression landmark times
time_A_prog = to_num(df.get("t_prog_from_60d_days"))
time_B_prog = to_num(df.get("t_prog_from_120d_days"))

# Model matrix (pre-treatment covariates)
use_cat = [c for c in ["age_recode","race_origin_recode","registry","grade_dcisset","er","pr","tumor_size_recode"]
           if c in df.columns]
X_cat = pd.get_dummies(df[use_cat], drop_first=True) if use_cat else pd.DataFrame(index=df.index)
X_num = pd.DataFrame()
if "dx_year" in df.columns:
    X_num["dx_year"] = to_num(df["dx_year"])
X = pd.concat([X_num, X_cat], axis=1).fillna(0.0)

HORIZONS = [5, 10, 15]
OUTCOMES = {
    "bcsm": {"event": event_bcsm, "Ycol": lambda h: f"Y_BCSM_{h}y", "kind": "pre"},
    "os":   {"event": event_any,  "Ycol": lambda h: f"Y_OS_{h}y",   "kind": "pre"},
    "prog": {"event": event_prog, "Ycol": None,                     "kind": "fly"},
}

# -------- Learner selection (auto) --------
ECONML_AVAILABLE = False
_DR, _DRF = None, None
if ENGINE in ("auto", "dr", "forestdr"):
    try:
        from econml.dr import DRLearner as _DR
        from econml.dr import ForestDRLearner as _DRF
        ECONML_AVAILABLE = True
    except Exception:
        ECONML_AVAILABLE = False

if ENGINE == "auto":
    ENGINE_EFF = "dr" if ECONML_AVAILABLE else "tlearner"
elif ENGINE in ("dr", "forestdr", "tlearner"):
    ENGINE_EFF = ENGINE if (ENGINE != "tlearner" or not ECONML_AVAILABLE) else "tlearner"
    if ENGINE_EFF != "tlearner" and not ECONML_AVAILABLE:
        ENGINE_EFF = "tlearner"
else:
    ENGINE_EFF = "tlearner"

print(f"[ENGINE] Effective learner: {ENGINE_EFF} (econml_available={ECONML_AVAILABLE})")

def fit_effects(Xf, Yf, Tf, Wf, seed):
    """Return tau = p1 - p0 predictions on Xf, or None if not available."""
    Yf = np.asarray(Yf).ravel()
    Tf = np.asarray(Tf).ravel()
    Wf = np.asarray(Wf).ravel()

    if ENGINE_EFF == "tlearner":
        Xmat = Xf.values if isinstance(Xf, pd.DataFrame) else np.asarray(Xf)
        if (Tf==0).sum()==0 or (Tf==1).sum()==0:
            return np.zeros(len(Xmat))
        m0 = GradientBoostingClassifier(random_state=seed+1)
        m1 = GradientBoostingClassifier(random_state=seed+2)
        m0.fit(Xmat[Tf==0], Yf[Tf==0], sample_weight=Wf[Tf==0])
        m1.fit(Xmat[Tf==1], Yf[Tf==1], sample_weight=Wf[Tf==1])
        p0 = m0.predict_proba(Xmat)[:,1]
        p1 = m1.predict_proba(Xmat)[:,1]
        return (p1 - p0).reshape(-1)

    if ENGINE_EFF == "forestdr":
        outcome = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        propens = LogisticRegression(max_iter=1000, solver="liblinear")
        try:
            learner = _DRF(model_regression=outcome, model_propensity=propens,
                           n_estimators=600, min_samples_leaf=10, min_samples_split=20,
                           random_state=seed, discrete_outcome=True, categories="auto", n_jobs=-1)
        except TypeError:
            learner = _DRF(model_regression=outcome, model_propensity=propens,
                           n_estimators=600, min_samples_leaf=10, min_samples_split=20,
                           random_state=seed, discrete_outcome=True, categories="auto")
        learner.fit(Yf, Tf, X=Xf, sample_weight=Wf)
        return np.asarray(learner.effect(Xf)).reshape(-1)

    # Default DR
    outcome = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    propens = LogisticRegression(max_iter=1000, solver="liblinear")
    final   = RandomForestRegressor(n_estimators=500, min_samples_leaf=10, random_state=seed+3, n_jobs=-1)
    learner = _DR(model_regression=outcome, model_propensity=propens,
                  model_final=final, discrete_outcome=True, random_state=seed)
    learner.fit(Yf, Tf, X=Xf, sample_weight=Wf)
    return np.asarray(learner.effect(Xf)).reshape(-1)

# =======================
# Contrast A: S vs OBS (t0 = 60d)
#   T=1: Surgery only (no RT) AND treated_by_t0
#   T=0: Observation (no surgery, no RT)
# =======================
S_only = (any_surgery & (~any_radiation))
OBS    = (~any_surgery & ~any_radiation)
T_SOBS_all = (S_only & treated_by_t0).astype(int)

timing_mask = (~unknown_timing) if EXCLUDE_UNKNOWN_TIMING else pd.Series(True, index=df.index)

for out_key, spec in OUTCOMES.items():
    ev = spec["event"]
    for H in HORIZONS:
        h_days = int(round(H*365.25))

        # Labels
        if spec["kind"] == "pre":
            Y_col = spec["Ycol"](H)
            if Y_col not in df.columns:
                print(f"[Skip] Missing {Y_col} in prepared file."); continue
            Y_all = pd.to_numeric(df[Y_col], errors="coerce").fillna(0).astype(int)
            time_used = time_A_bcsm_os
        else:
            # PROGRESSION: compute from landmarked time (60d)
            time_used = time_A_prog
            if time_used is None or (pd.isna(time_used).all()):
                print("[Skip] No progression timing (from 60d) found for S vs OBS."); continue
            Y_all = ((ev == 1) & (time_used <= h_days)).astype(int)

        # IPCW on eligible set
        W_raw_km, valid_km = ipcw_km(time_used, ev, h_days, group=T_SOBS_all)
        eligible = (S_only | OBS).to_numpy()
        combined = eligible & valid_km & timing_mask.to_numpy()
        valid_idx = df.index[combined]
        # align weights to valid set
        W_valid = W_raw_km[eligible[valid_km] & timing_mask.to_numpy()[valid_km]]

        # Propensity & trimming
        if len(valid_idx) == 0:
            print(f"[Skip] S vs OBS | {out_key.upper()} {H}y: no valid subjects."); continue
        gbm = GradientBoostingClassifier(random_state=100+H)
        gbm.fit(X.loc[valid_idx], T_SOBS_all.loc[valid_idx].to_numpy().ravel())
        p = gbm.predict_proba(X.loc[valid_idx])[:, 1]
        keep_np = ((p >= TRIM_LOW) & (p <= TRIM_HIGH)).astype(bool)

        # Assemble
        keep_idx = valid_idx[keep_np]
        if len(keep_idx) == 0:
            print(f"[Skip] S vs OBS | {out_key.upper()} {H}y: all trimmed."); continue
        XA = X.loc[keep_idx].copy()
        TA = T_SOBS_all.loc[keep_idx].to_numpy().ravel()
        YA = Y_all.loc[keep_idx].to_numpy().ravel()
        W  = W_valid[keep_np]

        # Save design (numeric-coerced)
        out_design = os.path.join(OUT_DIR, f"inputs_labels_s_vs_obs_{H}y_{out_key}.csv")
        D = XA.copy()
        D["row_id"] = keep_idx
        D[IDCOL]    = df.loc[keep_idx, IDCOL].astype(str).values
        D["T"] = TA; D["Y"] = YA; D["W"] = W
        D = coerce_design_dtypes(D, IDCOL)
        D.to_csv(out_design, index=False, float_format="%.6g")

        # CATE (numeric-coerced)
        tau = fit_effects(XA, YA, TA, W, seed=1200+H)
        if tau is not None:
            out_cate = os.path.join(OUT_DIR, f"cate_s_vs_obs_{H}y_{out_key}.csv")
            colname  = f"tau_{out_key.upper()}{H}_S_vs_OBS"
            C = pd.DataFrame({"row_id": np.asarray(keep_idx), IDCOL: df.loc[keep_idx, IDCOL].astype(str).values, colname: tau})
            C = coerce_cate_dtypes(C, IDCOL, colname)
            C.to_csv(out_cate, index=False, float_format="%.6g")
            print(f"[S vs OBS | {out_key.upper()} {H}y] unique tau:", len(np.unique(np.round(tau, 6))))
        else:
            print(f"[S vs OBS | {out_key.upper()} {H}y] CATE not computed (design saved).")

# =======================
# Contrast B: S+RT vs S (t0 = 120d)
#   T=1: S+RT
#   T=0: S only
# =======================
coB_mask = any_surgery
coB = df[coB_mask].copy()
time_B = time_B_all.loc[coB.index]   # 120d anchor (BCSM/OS)
T_B_all = to_bool(coB["any_radiation"]).astype(int)

for out_key, spec in OUTCOMES.items():
    if out_key in ("bcsm","os"):
        ev_B = (to_bool(coB["event_bcsm"]) if out_key == "bcsm" else to_bool(coB["event_any"])).astype(int)
        time_used = time_B
    else:
        ev_B = coB["event_prog"].fillna(0).astype(int)
        time_used = to_num(coB.get("t_prog_from_120d_days"))
        if time_used is None or time_used.isna().all():
            print("[Skip] No progression timing (from 120d) found for S+RT vs S."); continue

    for H in HORIZONS:
        h_days = int(round(H*365.25))
        YB_all = ((ev_B == 1) & (time_used <= h_days)).astype(int).to_numpy()

        # IPCW on surgery subset
        W_raw_km, valid_km = ipcw_km(time_used.to_numpy(), ev_B.to_numpy(), h_days, group=T_B_all)
        valid_idx = coB.index[valid_km]

        # Propensity & trimming
        if len(valid_idx) == 0:
            print(f"[Skip] S+RT vs S | {out_key.upper()} {H}y: no valid subjects."); continue
        XB_valid = X.loc[valid_idx]
        TB_valid = T_B_all.loc[valid_idx].to_numpy().ravel()
        YB_valid = YB_all[valid_km]

        gbm = GradientBoostingClassifier(random_state=2100+H)
        gbm.fit(XB_valid, TB_valid)
        p = gbm.predict_proba(XB_valid)[:, 1]
        keep_np = ((p >= TRIM_LOW) & (p <= TRIM_HIGH)).astype(bool)

        # Trimmed sets
        keep_idx = valid_idx[keep_np]
        if len(keep_idx) == 0:
            print(f"[Skip] S+RT vs S | {out_key.upper()} {H}y: all trimmed."); continue
        XBk = X.loc[keep_idx].copy()
        TBk = TB_valid[keep_np]
        YBk = YB_valid[keep_np]
        W   = W_raw_km[keep_np]

        # Save design (numeric-coerced)
        out_design = os.path.join(OUT_DIR, f"inputs_labels_srt_vs_s_{H}y_{out_key}.csv")
        D = XBk.copy()
        D["row_id"] = keep_idx
        D[IDCOL]    = df.loc[keep_idx, IDCOL].astype(str).values
        D["T"] = TBk; D["Y"] = YBk; D["W"] = W
        D = coerce_design_dtypes(D, IDCOL)
        D.to_csv(out_design, index=False, float_format="%.6g")

        # CATE (numeric-coerced)
        tau = fit_effects(XBk, YBk, TBk, W, seed=2200+H)
        if tau is not None:
            out_cate = os.path.join(OUT_DIR, f"cate_srt_vs_s_{H}y_{out_key}.csv")
            colname  = f"tau_{out_key.upper()}{H}_SplusRT_vs_S"
            C = pd.DataFrame({"row_id": np.asarray(keep_idx), IDCOL: df.loc[keep_idx, IDCOL].astype(str).values, colname: tau})
            C = coerce_cate_dtypes(C, IDCOL, colname)
            C.to_csv(out_cate, index=False, float_format="%.6g")
            print(f"[S+RT vs S | {out_key.upper()} {H}y] unique tau:", len(np.unique(np.round(tau, 6))))
        else:
            print(f"[S+RT vs S | {out_key.upper()} {H}y] CATE not computed (design saved).")

print("\nSTEP-01 (revised w/ progression) done. Files written to:", OUT_DIR)
