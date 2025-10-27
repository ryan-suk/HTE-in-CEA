# =======================
# STEP-02 (revised for 3-arm CEA + PROGRESSION):
#   Merge CATEs for S_vs_OBS and SplusRT_vs_S (BCSM, OS, PROG @ 5/10/15),
#   compute ARR, summaries & optional rules,
#   derive effect parameters (p0, p1, S0, S1, HR + joint HR over 5/10/15).
# =======================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier

# -------- Paths --------
BASE_DIR  = r"C:\Users\ryann\OneDrive\A_2025-26_Manuscripts\DCIS HTE\SEER files"
PATH_PREP = os.path.join(BASE_DIR, "dcis_prepared.csv")

prep = pd.read_csv(PATH_PREP, dtype=str).rename(columns=lambda c: c.strip())
prep = prep.reset_index().rename(columns={"index": "row_id"})
IDCOL = "patient_id" if "patient_id" in prep.columns else ("Patient ID" if "Patient ID" in prep.columns else None)
if IDCOL is None:
    raise ValueError("Need patient_id or Patient ID in prepared file.")
prep[IDCOL] = prep[IDCOL].astype(str)
prep["row_id"] = pd.to_numeric(prep["row_id"], errors="coerce").fillna(0).astype("int64")

HORIZONS  = [5, 10, 15]
OUTCOMES  = ["bcsm", "os", "prog"]   # <-- PROGRESSION added
CONTRASTS = {
    # mapping: tag -> pretty name + baseline label
    "s_vs_obs": {"pretty": "S_vs_OBS",     "baseline": "OBS (T=0)"},
    "srt_vs_s": {"pretty": "SplusRT_vs_S", "baseline": "Surgery (T=0)"},
}

# ---------- Helpers ----------
def load_cate_safe(path, tau_col_expected):
    """
    Load CATE file robustly and return [row_id, IDCOL, tau_col_expected] as numeric.
    If the actual tau column name differs but there's exactly one tau_* column, rename it.
    """
    if (path is None) or (not os.path.exists(path)):
        return pd.DataFrame(columns=["row_id", IDCOL, tau_col_expected])

    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"'row_id' missing in {path}")

    if IDCOL not in df.columns:
        df = df.merge(prep[["row_id", IDCOL]], on="row_id", how="left")

    # find tau column
    tau_cols = [c for c in df.columns if c.lower().startswith("tau_")]
    if tau_col_expected not in df.columns:
        if len(tau_cols) == 1:
            df = df.rename(columns={tau_cols[0]: tau_col_expected})
        else:
            # cannot resolve—return empty
            return pd.DataFrame(columns=["row_id", IDCOL, tau_col_expected])

    out = df[["row_id", IDCOL, tau_col_expected]].copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce").astype("Int64")
    out[IDCOL]    = out[IDCOL].astype(str)
    out[tau_col_expected] = pd.to_numeric(out[tau_col_expected], errors="coerce").astype("float64")
    return out

def load_inputs_labels(path):
    """
    Load inputs/labels design file -> (df, feature_cols).
    Feature columns here are already numeric (one-hot) from STEP-01.
    """
    if (path is None) or (not os.path.exists(path)):
        return None, []
    df = pd.read_csv(path)
    required = {"row_id", IDCOL, "T", "Y", "W"}
    if not required.issubset(df.columns):
        missing = required.difference(set(df.columns))
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    feat_cols = [c for c in df.columns if c not in required]
    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(0).astype("int64")
    df[IDCOL]    = df[IDCOL].astype(str)
    df["T"]      = pd.to_numeric(df["T"], errors="coerce").fillna(0).astype("int8")
    df["Y"]      = pd.to_numeric(df["Y"], errors="coerce").fillna(0).astype("int8")
    df["W"]      = pd.to_numeric(df["W"], errors="coerce").fillna(1.0).astype("float64")
    # coerce features numeric (defensive)
    for f in feat_cols:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0).astype("float64")
    return df, feat_cols

def safe_cat(s):
    return s.astype(str).fillna("Unknown")

# ---------- Merge CATEs → ARR (for bcsm, os, prog) ----------
merged = prep.copy()
ARR_COLS = []
TAU_COLS = []

for outcome in OUTCOMES:
    for H in HORIZONS:
        for tag, meta in CONTRASTS.items():
            pretty = meta["pretty"]
            cate_path = os.path.join(BASE_DIR, f"cate_{tag}_{H}y_{outcome}.csv")
            tau_col   = f"tau_{outcome.upper()}{H}_{pretty}"
            cate = load_cate_safe(cate_path, tau_col)
            merged = merged.merge(cate, on=["row_id", IDCOL], how="left")

            # ARR = -tau (since tau = p1 - p0 for event-risk; ARR = p0 - p1)
            arr_col = f"ARR_{outcome.upper()}{H}_{pretty}"
            if tau_col in merged.columns:
                merged[tau_col] = pd.to_numeric(merged[tau_col], errors="coerce").astype("float64")
                merged[arr_col] = -merged[tau_col]
                ARR_COLS.append(arr_col)
                TAU_COLS.append(tau_col)

# ensure numeric formatting where applicable
for col in ARR_COLS + TAU_COLS:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

OUT_WIDE = os.path.join(BASE_DIR, "prepared_with_cates_wide.csv")
merged.to_csv(OUT_WIDE, index=False, float_format="%.6g")
print("Saved:", OUT_WIDE)

# ---------- Long format (ARR only) ----------
long_parts = []
for outcome in OUTCOMES:
    for H in HORIZONS:
        for tag, meta in CONTRASTS.items():
            pretty  = meta["pretty"]
            arr_col = f"ARR_{outcome.upper()}{H}_{pretty}"
            if arr_col in merged.columns:
                t = merged[["row_id", IDCOL, arr_col]].rename(columns={arr_col: "ARR"})
                t["ARR"] = pd.to_numeric(t["ARR"], errors="coerce")
                t["contrast"] = f"{pretty}_{outcome.upper()}_{H}y"
                long_parts.append(t)

if long_parts:
    OUT_LONG = os.path.join(BASE_DIR, "prepared_with_cates_long.csv")
    pd.concat(long_parts, ignore_index=True).to_csv(OUT_LONG, index=False, float_format="%.6g")
    print("Saved:", OUT_LONG)

# ---------- Summaries by features ----------
COLS = {
    "age":      "age_recode"         if "age_recode" in merged.columns else None,
    "grade":    "grade_dcisset"      if "grade_dcisset" in merged.columns else None,
    "er":       "er"                 if "er" in merged.columns else None,
    "pr":       "pr"                 if "pr" in merged.columns else None,
    "size":     "tumor_size_recode"  if "tumor_size_recode" in merged.columns else None,
    "registry": "registry"           if "registry" in merged.columns else None,
    "race":     "race_origin_recode" if "race_origin_recode" in merged.columns else None,
    "year":     "dx_year"            if "dx_year" in merged.columns else None,
}

def summary_by(feature_col, arr_col, out_path):
    df2 = merged[[feature_col, arr_col]].copy()
    df2[feature_col] = safe_cat(df2[feature_col])
    df2[arr_col]     = pd.to_numeric(df2[arr_col], errors="coerce")
    out = (df2.groupby(feature_col, dropna=False)[arr_col]
             .agg(n="count", mean="mean", sd="std",
                  q25=lambda x: x.quantile(0.25), q50="median", q75=lambda x: x.quantile(0.75))
             .reset_index()
             .sort_values("mean", ascending=False))
    out.to_csv(out_path, index=False, float_format="%.6g")

for arr_col in ARR_COLS:
    for key, feat in COLS.items():
        if not feat:
            continue
        outp = os.path.join(BASE_DIR, f"summary_{arr_col}_by_{key}.csv")
        summary_by(feat, arr_col, outp)

# ---------- OPTIONAL: small CART rules on ARR ----------
def make_rules(arr_col, outprefix):
    if arr_col not in merged.columns:
        return
    Z = pd.DataFrame({
        "age":  safe_cat(merged.get(COLS["age"],    pd.Series("Unknown", index=merged.index))),
        "size": safe_cat(merged.get(COLS["size"],   pd.Series("Unknown", index=merged.index))),
        "grade":safe_cat(merged.get(COLS["grade"],  pd.Series("Unknown", index=merged.index))),
        "ER":   safe_cat(merged.get(COLS["er"],     pd.Series("Unknown", index=merged.index))),
        "PR":   safe_cat(merged.get(COLS["pr"],     pd.Series("Unknown", index=merged.index))),
        "year": safe_cat(merged.get(COLS["year"],   pd.Series("Unknown", index=merged.index))),
        "race": safe_cat(merged.get(COLS["race"],   pd.Series("Unknown", index=merged.index))),
        "reg":  safe_cat(merged.get(COLS["registry"], pd.Series("Unknown", index=merged.index))),
    })
    Z_enc = pd.get_dummies(Z, drop_first=True)
    y = pd.to_numeric(merged[arr_col], errors="coerce")
    keep = y.notna()
    if keep.sum() < 1000:
        # too few to fit a stable tree; skip
        return
    Z_tr, Z_ev, y_tr, y_ev = train_test_split(Z_enc[keep], y[keep], test_size=0.4, random_state=42)
    gs = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {"max_depth":[1,2,3], "min_samples_leaf":[200,400,800]},
        scoring="neg_mean_squared_error", cv=5
    )
    gs.fit(Z_tr, y_tr)
    tree = gs.best_estimator_
    txt = export_text(tree, feature_names=list(Z_tr.columns))
    with open(os.path.join(BASE_DIR, f"{outprefix}_tree.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

    leaves = tree.apply(Z_ev)
    rows=[]
    for lid in np.unique(leaves):
        idx = (leaves == lid); n = int(np.sum(idx))
        if n == 0:
            continue
        mu = float(np.mean(y_ev.iloc[idx]))
        boot = [np.mean(resample(y_ev.iloc[idx], replace=True, n_samples=n, random_state=i))
                for i in range(400)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"leaf_id": lid, "n": n, "mean_ARR": mu, "ci95_lo": lo, "ci95_hi": hi})
    pd.DataFrame(rows).sort_values("mean_ARR", ascending=False)\
      .to_csv(os.path.join(BASE_DIR, f"{outprefix}_subgroup_leaves.csv"), index=False, float_format="%.6g")

# Example (uncomment to write rule artifacts)
# for outcome in OUTCOMES:
#     for H in HORIZONS:
#         for tag, meta in CONTRASTS.items():
#             pretty  = meta["pretty"]
#             arr_col = f"ARR_{outcome.upper()}{H}_{pretty}"
#             make_rules(arr_col, f"rules_{outcome.upper()}_{pretty}_{H}y")

# ---------- Effect parameters for CEA ----------
def effect_params_from_design(design_path, cate_path, tau_col, horizon_tag, comparator_label, suffix):
    """
    For one contrast/outcome/horizon, derive p0, p1, S0, S1, HR per patient.
    p0 is learned on comparator (T==0) using GBM with IPCW weights if present.
    """
    IL, feats = load_inputs_labels(design_path)
    if (IL is None) or (not os.path.exists(cate_path)):
        return pd.DataFrame(columns=[
            "row_id", IDCOL,
            f"p0_{horizon_tag}_{suffix}", f"p1_{horizon_tag}_{suffix}",
            f"S0_{horizon_tag}_{suffix}", f"S1_{horizon_tag}_{suffix}",
            f"HR_{horizon_tag}_{suffix}", "baseline_arm"
        ])

    cate = load_cate_safe(cate_path, tau_col)
    if cate.empty:
        return pd.DataFrame(columns=[
            "row_id", IDCOL,
            f"p0_{horizon_tag}_{suffix}", f"p1_{horizon_tag}_{suffix}",
            f"S0_{horizon_tag}_{suffix}", f"S1_{horizon_tag}_{suffix}",
            f"HR_{horizon_tag}_{suffix}", "baseline_arm"
        ])

    # Fit p0 on comparator rows
    cmp_mask = (IL["T"] == 0)
    if cmp_mask.sum() == 0:
        return pd.DataFrame(columns=[
            "row_id", IDCOL,
            f"p0_{horizon_tag}_{suffix}", f"p1_{horizon_tag}_{suffix}",
            f"S0_{horizon_tag}_{suffix}", f"S1_{horizon_tag}_{suffix}",
            f"HR_{horizon_tag}_{suffix}", "baseline_arm"
        ])
    base = GradientBoostingClassifier(random_state={"5y":101,"10y":201,"15y":301}.get(horizon_tag, 999))
    try:
        base.fit(IL.loc[cmp_mask, feats], IL.loc[cmp_mask,"Y"].to_numpy().ravel(),
                 sample_weight=IL.loc[cmp_mask,"W"].to_numpy())
    except Exception:
        base.fit(IL.loc[cmp_mask, feats], IL.loc[cmp_mask,"Y"].to_numpy().ravel())

    # Merge tau with features on patients present in both
    C = cate.merge(IL[["row_id", IDCOL] + feats], on=["row_id", IDCOL], how="inner")
    if C.empty:
        return pd.DataFrame(columns=[
            "row_id", IDCOL,
            f"p0_{horizon_tag}_{suffix}", f"p1_{horizon_tag}_{suffix}",
            f"S0_{horizon_tag}_{suffix}", f"S1_{horizon_tag}_{suffix}",
            f"HR_{horizon_tag}_{suffix}", "baseline_arm"
        ])

    p0  = base.predict_proba(C[feats])[:, 1]
    tau = pd.to_numeric(C[tau_col], errors="coerce").to_numpy()

    eps = 1e-6
    p0 = np.clip(p0, eps, 1-eps)
    p1 = np.clip(p0 + tau, eps, 1-eps)   # tau = p1 - p0
    S0 = np.clip(1 - p0, eps, 1-eps)
    S1 = np.clip(1 - p1, eps, 1-eps)
    HR = np.log(S1) / np.log(S0)

    out = pd.DataFrame({
        "row_id": C["row_id"].to_numpy(),
        IDCOL:    C[IDCOL].astype(str).to_numpy(),
        f"p0_{horizon_tag}_{suffix}": p0.astype("float64"),
        f"p1_{horizon_tag}_{suffix}": p1.astype("float64"),
        f"S0_{horizon_tag}_{suffix}": S0.astype("float64"),
        f"S1_{horizon_tag}_{suffix}": S1.astype("float64"),
        f"HR_{horizon_tag}_{suffix}": HR.astype("float64"),
        "baseline_arm": np.array([comparator_label]*len(C), dtype=object)
    })
    return out

def derive_effect_params_multi(tag, pretty_label, outcome, baseline_label, suffix):
    """
    Assemble across 5/10/15y for one contrast/outcome and write CSV with HR_joint.
    """
    tables = []
    for H in HORIZONS:
        design  = os.path.join(BASE_DIR, f"inputs_labels_{tag}_{H}y_{outcome}.csv")
        cate    = os.path.join(BASE_DIR, f"cate_{tag}_{H}y_{outcome}.csv")
        tau_col = f"tau_{outcome.upper()}{H}_{pretty_label}"
        tstr    = f"{H}y"
        tables.append(
            effect_params_from_design(design, cate, tau_col, tstr, baseline_label, suffix)
        )

    nonempty = [t for t in tables if (t is not None) and (not t.empty)]
    if not nonempty:
        print(f"[Info] No effect params for {tag} ({outcome}).")
        return

    eff = nonempty[0]
    for t in nonempty[1:]:
        eff = eff.merge(t, on=["row_id", IDCOL, "baseline_arm"], how="outer")

    # Joint HR across available horizons by least squares on log-survivals
    eps = 1e-12
    HR_joint = np.full(len(eff), np.nan, float)
    num = np.zeros(len(eff)); den = np.zeros(len(eff))
    for H in HORIZONS:
        s0 = eff.get(f"S0_{H}y_{suffix}")
        s1 = eff.get(f"S1_{H}y_{suffix}")
        if s0 is None or s1 is None:
            continue
        mask = s0.notna() & s1.notna()
        if mask.any():
            l0 = np.log(np.clip(s0[mask].to_numpy(), eps, 1-eps))
            l1 = np.log(np.clip(s1[mask].to_numpy(), eps, 1-eps))
            idx = np.where(mask.to_numpy())[0]
            num[idx] += (l0 * l1); den[idx] += (l0 ** 2)
    good = den > 0
    HR_joint[good] = num[good] / den[good]
    eff[f"HR_joint_{suffix}"] = HR_joint

    # numeric-safe write
    eff["row_id"] = pd.to_numeric(eff["row_id"], errors="coerce").fillna(0).astype("int64")
    eff.to_csv(os.path.join(BASE_DIR, f"effect_params_{tag}_{suffix}.csv"), index=False, float_format="%.6g")
    print("Saved effect parameters ->", os.path.join(BASE_DIR, f"effect_params_{tag}_{suffix}.csv"))

# ---------- Build effect parameters for both contrasts & all outcomes ----------
# S vs OBS
derive_effect_params_multi(tag="s_vs_obs", pretty_label="S_vs_OBS",
                           outcome="bcsm", baseline_label=CONTRASTS["s_vs_obs"]["baseline"], suffix="bcsm")
derive_effect_params_multi(tag="s_vs_obs", pretty_label="S_vs_OBS",
                           outcome="os",   baseline_label=CONTRASTS["s_vs_obs"]["baseline"], suffix="os")
derive_effect_params_multi(tag="s_vs_obs", pretty_label="S_vs_OBS",
                           outcome="prog", baseline_label=CONTRASTS["s_vs_obs"]["baseline"], suffix="prog")

# S+RT vs S
derive_effect_params_multi(tag="srt_vs_s", pretty_label="SplusRT_vs_S",
                           outcome="bcsm", baseline_label=CONTRASTS["srt_vs_s"]["baseline"], suffix="bcsm")
derive_effect_params_multi(tag="srt_vs_s", pretty_label="SplusRT_vs_S",
                           outcome="os",   baseline_label=CONTRASTS["srt_vs_s"]["baseline"], suffix="os")
derive_effect_params_multi(tag="srt_vs_s", pretty_label="SplusRT_vs_S",
                           outcome="prog", baseline_label=CONTRASTS["srt_vs_s"]["baseline"], suffix="prog")

print("\nSTEP-02 (revised, incl. PROGRESSION) done.")
