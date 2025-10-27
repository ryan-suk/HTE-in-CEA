# =======================
# STEP-04 (revised): Top-K policy-tree (no dx_year) + SSR subgroups,
# then 3-arm CEA microsimulation (OBS, S, S+RT) with PROGRESSION state
# - Objective for subgrouping: ARR_PROG10_SplusRT_vs_S (10y progression ARR)
# - Honest splitting; CV-pruning; minimum honest leaf size
# - SSR: unions of best leaves; pick up to K disjoint subgroups
# - Microsim states: Alive (no progression), Alive (progressed), Dead (BCSM / Other)
# - Progression hazards per arm via per-patient PH multipliers
# - Outputs overall & subgroup CEAs (patient-level, CE-probs, pairwise ICER, frontier)
# =======================

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_text
from lifelines import WeibullFitter, LogLogisticFitter, LogNormalFitter

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------- Paths --------
BASE_DIR   = r"C:\Users\ryann\OneDrive\A_2025-26_Manuscripts\DCIS HTE\SEER files"
PATH_PREP  = os.path.join(BASE_DIR, "dcis_prepared.csv")
PATH_WIDE  = os.path.join(BASE_DIR, "prepared_with_cates_wide.csv")

# Per-patient θ tables (if present). If not, we fall back to STEP-02 effect-params
PATH_HR_BCSM_S_OBS  = os.path.join(BASE_DIR, "hr_s_vs_obs_by_patient.csv")
PATH_HR_BCSM_SRT_S  = os.path.join(BASE_DIR, "hr_splusrt_vs_s_by_patient.csv")
PATH_HRPROG_S_OBS   = os.path.join(BASE_DIR, "hr_prog_s_vs_obs_by_patient.csv")
PATH_HRPROG_SRT_S   = os.path.join(BASE_DIR, "hr_prog_splusrt_vs_s_by_patient.csv")

# STEP-02 effect-parameter files (fallback for θ)
PATH_EFF_BCSM_S_OBS = os.path.join(BASE_DIR, "effect_params_s_vs_obs_bcsm.csv")
PATH_EFF_BCSM_SRT_S = os.path.join(BASE_DIR, "effect_params_srt_vs_s_bcsm.csv")
PATH_EFF_PROG_S_OBS = os.path.join(BASE_DIR, "effect_params_s_vs_obs_prog.csv")
PATH_EFF_PROG_SRT_S = os.path.join(BASE_DIR, "effect_params_srt_vs_s_prog.csv")

# Optional raw progression file (to merge into PREP if missing)
PATH_PROG = os.path.join(BASE_DIR, "dcis_progression.csv")

# -------- Toggles / hyperparameters --------
np.random.seed(20251025)

IDCOL_CANDIDATES = ["patient_id", "Patient ID"]
OBJECTIVE_ARR = "ARR_PROG10_SplusRT_vs_S"   # Progression ARR at 10y for policy tree & SSR

# Honest tree
TRAIN_FRAC = 0.5
TREE_PARAM_GRID = {"max_depth": [2, 3, 4], "min_samples_leaf": [200, 400, 800]}
MIN_HONEST_LEAF = 150

# SSR search
SSR_MIN_N = 300           # minimum subgroup size on HONEST sample
SSR_TOP_K = 3             # choose up to K disjoint subgroups
SSR_MAX_LEAVES = 10       # consider only top-M leaves by honest mean objective

# Microsimulation options
DISCOUNT_RATE_ANNUAL = 0.03
DT_DAYS  = 30.4375
MAX_YEARS = 40
MIN_S_BASE_N = 200        # need enough S-only in subgroup to fit subgroup-specific baseline; else fallback to overall S-only

# -------- Economic placeholders --------
COSTS = dict(
    C_OBS_INIT           = 0.0,
    C_S_INIT             = 8000.0,
    C_RT_COURSE          = 8000.0,
    C_MONTHLY_FOLLOWUP   = 50.0,
    C_PROG_INIT          = 2000.0,    # one-off cost at progression
    C_PROG_MONTHLY       = 150.0,     # monthly cost after progression while alive
    C_EOL_BCSM           = 12000.0,
    C_EOL_OTHER          = 8000.0
)
UTILS = dict(
    U_ALIVE              = 0.83,  # baseline utility (no progression)
    DU_PROG_CHRONIC      = 0.05,  # chronic disutility after progression
    DU_S_ACUTE           = 0.01,
    S_ACUTE_MONTHS       = 1,
    DU_RT_ACUTE          = 0.02,
    RT_ACUTE_MONTHS      = 2
)

# -------- Helpers --------
def to_num(s):  return pd.to_numeric(s, errors="coerce")
def to_bool(s): return s.astype(str).str.strip().str.lower().isin(["true","1","t","y","yes"])

def find_idcol(df):
    for c in IDCOL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No patient ID column found (tried: {}).".format(IDCOL_CANDIDATES))

def ensure_str_id(df, idcol):
    df[idcol] = df[idcol].astype(str)
    return df

# ----- Load PREP and WIDE
prep = pd.read_csv(PATH_PREP, dtype=str).rename(columns=str.strip)
IDCOL = find_idcol(prep)
prep  = ensure_str_id(prep, IDCOL)

wide = pd.read_csv(PATH_WIDE, dtype=str).rename(columns=str.strip)
wide = ensure_str_id(wide, IDCOL)

# ----- Merge progression columns into PREP (robust)
if ("event_prog" not in prep.columns) or ("t_prog_from_120d_days" not in prep.columns):
    if os.path.exists(PATH_PROG):
        prog = pd.read_csv(PATH_PROG, dtype=str).rename(columns=str.strip)
        # align ID
        if IDCOL not in prog.columns:
            for c in IDCOL_CANDIDATES:
                if c in prog.columns:
                    prog = prog.rename(columns={c: IDCOL})
                    break
        prog = ensure_str_id(prog, IDCOL)

        # event column
        evt_candidates = ["event_prog","event_progression","progression_event","prog_event"]
        evt_col = next((c for c in evt_candidates if c in prog.columns), None)
        if evt_col is not None:
            prog["event_prog"] = prog[evt_col].astype(str).str.strip().str.lower()\
                                     .isin(["1","true","t","yes","y"]).astype(int)
        else:
            prog["event_prog"] = 0

        # timing from 120d
        if "t_prog_from_120d_days" in prog.columns:
            prog["t_prog_from_120d_days"] = pd.to_numeric(prog["t_prog_from_120d_days"], errors="coerce")
        elif "prog_days_from_dx" in prog.columns:
            prog["t_prog_from_120d_days"] = pd.to_numeric(prog["prog_days_from_dx"], errors="coerce") - 120.0
        elif "t_prog_from_60d_days" in prog.columns:
            prog["t_prog_from_120d_days"] = pd.to_numeric(prog["t_prog_from_60d_days"], errors="coerce") - 60.0
        else:
            prog["t_prog_from_120d_days"] = np.nan

        prep = prep.merge(
            prog[[IDCOL, "event_prog", "t_prog_from_120d_days"]].drop_duplicates(subset=[IDCOL]),
            on=IDCOL, how="left"
        )
        print("[Merge] Progression merged into PREP.")
    else:
        warnings.warn("Progression columns not in PREP and PATH_PROG not found; "
                      "progression baseline will use flat tiny hazard fallback.")

# ----- Merge ARR objective onto PREP
if OBJECTIVE_ARR not in wide.columns:
    raise ValueError(f"{OBJECTIVE_ARR} not found in {PATH_WIDE}. Run STEP-02 first.")
DF = prep.merge(wide[[IDCOL, OBJECTIVE_ARR]], on=IDCOL, how="left")
DF[OBJECTIVE_ARR] = pd.to_numeric(DF[OBJECTIVE_ARR], errors="coerce")

# ----- Load θ tables (BCSM & PROG), with fallback to effect-params
def load_theta_or_fallback(hr_path, eff_path, eff_col_name, outcol):
    """
    Try to read a per-patient HR table (hr_path), accepting several candidate theta columns.
    If not present, fall back to STEP-02 effect-params (eff_col_name).
    Returns a DataFrame: [IDCOL, outcol]
    """
    # 1) Attempt HR table
    if os.path.exists(hr_path):
        hr = pd.read_csv(hr_path, dtype=str).rename(columns=str.strip)
        # resolve ID
        hr_id = find_idcol(hr) if any(c in hr.columns for c in IDCOL_CANDIDATES) else None
        if hr_id is None and IDCOL in hr.columns:
            hr_id = IDCOL
        if hr_id is None:
            # no ID in HR table -> skip to fallback
            pass
        else:
            hr = ensure_str_id(hr, hr_id)
            cand_cols = [
                "theta_use", outcol, "theta", "theta_bcsm", "theta_prog",
                "theta_bcsm_S_vs_OBS", "theta_bcsm_SRT_vs_S",
                "theta_prog_S_vs_OBS", "theta_prog_SRT_vs_S"
            ]
            cand_cols = [c for c in cand_cols if c in hr.columns]
            if len(cand_cols) > 0:
                # prefer exact outcol, else first available
                csel = outcol if outcol in cand_cols else cand_cols[0]
                tmp = hr[[hr_id, csel]].copy()
                tmp = tmp.rename(columns={hr_id: IDCOL, csel: outcol})
                tmp[outcol] = pd.to_numeric(tmp[outcol], errors="coerce")
                print(f"[θ] Using '{csel}' from {os.path.basename(hr_path)} -> {outcol}")
                return tmp[[IDCOL, outcol]]

    # 2) Fallback to effect-parameters
    if os.path.exists(eff_path):
        eff = pd.read_csv(eff_path, dtype=str).rename(columns=str.strip)
        if IDCOL not in eff.columns:
            eff_id = find_idcol(eff)
            eff = eff.rename(columns={eff_id: IDCOL})
        eff = ensure_str_id(eff, IDCOL)
        if eff_col_name not in eff.columns:
            raise ValueError(f"Fallback effect-params file {eff_path} lacks '{eff_col_name}'.")
        out = eff[[IDCOL, eff_col_name]].copy()
        out[outcol] = pd.to_numeric(out[eff_col_name], errors="coerce")
        out = out.drop(columns=[eff_col_name])
        print(f"[θ] Using '{eff_col_name}' from {os.path.basename(eff_path)} -> {outcol}")
        return out[[IDCOL, outcol]]

    raise ValueError(f"Neither HR table nor effect-params available for {outcol}.")

ThetaB_S_OBS = load_theta_or_fallback(PATH_HR_BCSM_S_OBS,  PATH_EFF_BCSM_S_OBS, "HR_joint_bcsm", "theta_bcsm_S_vs_OBS")
ThetaB_SRT_S = load_theta_or_fallback(PATH_HR_BCSM_SRT_S,  PATH_EFF_BCSM_SRT_S, "HR_joint_bcsm", "theta_bcsm_SRT_vs_S")
ThetaP_S_OBS = load_theta_or_fallback(PATH_HRPROG_S_OBS,   PATH_EFF_PROG_S_OBS,  "HR_joint_prog", "theta_prog_S_vs_OBS")
ThetaP_SRT_S = load_theta_or_fallback(PATH_HRPROG_SRT_S,   PATH_EFF_PROG_SRT_S,  "HR_joint_prog", "theta_prog_SRT_vs_S")

# Merge θ onto DF
DF = (DF.merge(ThetaB_S_OBS, on=IDCOL, how="left")
        .merge(ThetaB_SRT_S, on=IDCOL, how="left")
        .merge(ThetaP_S_OBS, on=IDCOL, how="left")
        .merge(ThetaP_SRT_S, on=IDCOL, how="left"))

# Combined θ vs OBS for S+RT (not strictly needed, but useful to export/inspect)
DF["theta_bcsm_SRT_vs_OBS"] = DF["theta_bcsm_S_vs_OBS"] * DF["theta_bcsm_SRT_vs_S"]
DF["theta_prog_SRT_vs_OBS"] = DF["theta_prog_S_vs_OBS"] * DF["theta_prog_SRT_vs_S"]

print("[θ] Non-missing counts:",
      "bcsm S_vs_OBS =", DF["theta_bcsm_S_vs_OBS"].notna().sum(), "|",
      "bcsm SRT_vs_S =", DF["theta_bcsm_SRT_vs_S"].notna().sum(), "|",
      "prog S_vs_OBS =", DF["theta_prog_S_vs_OBS"].notna().sum(), "|",
      "prog SRT_vs_S =", DF["theta_prog_SRT_vs_S"].notna().sum())

# ----- Features for policy tree (interpretable) — EXCLUDE dx_year
FEATURES = [c for c in [
    "age_recode", "tumor_size_recode", "grade_dcisset",
    "race_origin_recode", "er", "pr", "registry" # <-- dx_year intentionally excluded
] if c in DF.columns]

# One-hot encode categoricals (keep readable categories via export_text)
X_cat = pd.get_dummies(DF[FEATURES], drop_first=True) if FEATURES else pd.DataFrame(index=DF.index)
y = DF[OBJECTIVE_ARR]
keep = y.notna()
X  = X_cat.loc[keep].copy()
y  = y.loc[keep].copy()
ID = DF.loc[keep, IDCOL].copy()

# ----- Honest split
X_tr, X_ho, y_tr, y_ho, id_tr, id_ho = train_test_split(
    X, y, ID, test_size=(1-TRAIN_FRAC), random_state=42, stratify=None
)

# ----- Fit small interpretable tree with CV pruning
gs = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    TREE_PARAM_GRID, scoring="neg_mean_squared_error", cv=5
)
gs.fit(X_tr, y_tr)
tree = gs.best_estimator_

# Enforce honest minimum leaf size
leaf_ho = tree.apply(X_ho)
leaf_counts = pd.Series(leaf_ho).value_counts()
valid_leaves = set(leaf_counts[leaf_counts >= MIN_HONEST_LEAF].index)
mask_valid = np.isin(leaf_ho, list(valid_leaves))

X_ho_v = X_ho[mask_valid].reset_index(drop=True)
y_ho_v = pd.Series(y_ho[mask_valid].to_numpy(), name=OBJECTIVE_ARR).reset_index(drop=True)
id_ho_v= id_ho[mask_valid].reset_index(drop=True)
leaf_ho_v = np.asarray(tree.apply(X_ho_v))

# Save readable tree
tree_txt = export_text(tree, feature_names=list(X.columns))
TREE_PATH = os.path.join(BASE_DIR, f"policy_tree__{OBJECTIVE_ARR}.txt")
with open(TREE_PATH, "w", encoding="utf-8") as f:
    f.write(tree_txt)
print(f"[Tree] Saved: {TREE_PATH}")

# ----- Compute honest leaf stats
leaf_tbl = (
    pd.DataFrame({"leaf": leaf_ho_v, "OBJ": y_ho_v.values})
      .groupby("leaf").agg(n=("OBJ","size"), mean_obj=("OBJ","mean"))
      .reset_index()
      .sort_values("mean_obj", ascending=False)
)
leaf_tbl_top = leaf_tbl.head(min(SSR_MAX_LEAVES, len(leaf_tbl))).copy()

# Map honest IDs into leaves
assign = pd.DataFrame({IDCOL: id_ho_v.values, "leaf": leaf_ho_v})
assign = assign[assign["leaf"].isin(leaf_tbl_top["leaf"])]

# ----- SSR: choose up to K disjoint unions of leaves (greedy)
selected_groups = []
remaining_ids = set(assign[IDCOL].unique())

for k in range(SSR_TOP_K):
    best = None
    for start_leaf in leaf_tbl_top["leaf"].tolist():
        if best is not None:
            break
        candidate = {start_leaf}
        members = assign[assign["leaf"].isin(candidate)][IDCOL].tolist()
        members = [m for m in members if m in remaining_ids]
        if len(members) < SSR_MIN_N:
            for add_leaf in leaf_tbl_top["leaf"].tolist():
                if add_leaf in candidate: continue
                more = assign[assign["leaf"]==add_leaf][IDCOL].tolist()
                more = [m for m in more if m in remaining_ids]
                cand_members = list(set(members) | set(more))
                if len(cand_members) >= SSR_MIN_N:
                    candidate.add(add_leaf)
                    members = cand_members
                    break
        if len(members) >= SSR_MIN_N:
            sel_idx = assign[assign[IDCOL].isin(members)].index
            mean_obj = float(y_ho_v.iloc[sel_idx].mean())
            best = dict(leaves=candidate, n=len(members), mean_obj=mean_obj, members=members)
    if best is None:
        print(f"[SSR] No more non-overlapping groups meeting size >= {SSR_MIN_N}.")
        break
    tag = f"Top{k+1}"
    selected_groups.append({"tag": tag, **best})
    remaining_ids -= set(best["members"])
    print(f"[SSR] Selected {tag}: n={best['n']}, mean ARR={best['mean_obj']:.4f}, leaves={sorted(list(best['leaves']))}")

# Always include an Overall cohort covering everyone in DF (for overall CEA)
selected_groups = [{"tag": "Overall", "leaves": None, "n": int(len(DF)),
                    "mean_obj": float(DF[OBJECTIVE_ARR].mean()),
                    "members": DF[IDCOL].tolist()}] + selected_groups

# Persist SSR selections
for g in selected_groups:
    gjson = {
        "objective": OBJECTIVE_ARR, "tag": g["tag"], "n": g["n"], "mean_obj": g["mean_obj"],
        "leaves": sorted(list(g["leaves"])) if g["leaves"] is not None else "ALL",
    }
    json_path = os.path.join(BASE_DIR, f"ssr_best_subgroup__{OBJECTIVE_ARR}__{g['tag']}.json")
    with open(json_path, "w") as f:
        json.dump(gjson, f, indent=2)
    mem_path = os.path.join(BASE_DIR, f"subgroup_members__{g['tag']}.csv")
    pd.DataFrame({IDCOL: list(dict.fromkeys(g["members"]))}).to_csv(mem_path, index=False)
    print(f"[SSR] Saved subgroup {g['tag']} (n={g['n']}) -> {mem_path}")

pd.DataFrame([{"tag": g["tag"], "n": g["n"], "mean_obj": g["mean_obj"]} for g in selected_groups])\
  .to_csv(os.path.join(BASE_DIR, "ssr_topk_summary.csv"), index=False)

# ---------------------------------------------------------
# Export human-readable rules
# ---------------------------------------------------------
def build_ohe_maps(prep_df, orig_features, X_cols):
    var_to_cats = {}
    dummy_to_pair = {}
    for var in orig_features:
        if var not in prep_df.columns:
            continue
        cats = sorted(prep_df[var].astype(str).fillna("Unknown").unique().tolist())
        prefix = var + "_"
        var_cols = [c for c in X_cols if c.startswith(prefix)]
        cats_with_cols = [c[len(prefix):] for c in var_cols]
        base_candidates = [c for c in cats if c not in cats_with_cols]
        base_cat = base_candidates[0] if base_candidates else None
        var_to_cats[var] = {"all": cats, "with_cols": cats_with_cols, "base": base_cat, "dummy_cols": var_cols}
        for dc, cat in zip(var_cols, cats_with_cols):
            dummy_to_pair[dc] = (var, cat)
    return var_to_cats, dummy_to_pair

def tree_leaf_paths(dt, feature_names):
    left = dt.tree_.children_left
    right= dt.tree_.children_right
    feat = dt.tree_.feature
    thr  = dt.tree_.threshold
    paths = {}
    stack = [(0, [])]
    while stack:
        node, conds = stack.pop()
        if left[node] == right[node] == -1:
            paths[node] = conds.copy()
        else:
            f = feature_names[feat[node]]; t = thr[node]
            stack.append((right[node], conds + [(f, ">", t)]))
            stack.append((left[node],  conds + [(f, "<=", t)]))
    return paths

def conditions_to_rule(var_to_cats, dummy_to_pair, conds):
    by_var = {var: {"forced": None, "exclude": set()} for var in var_to_cats.keys()}
    for fname, op, t in conds:
        if fname in dummy_to_pair:
            var, cat = dummy_to_pair[fname]
            if op == ">" and t < 0.5:
                by_var[var]["forced"] = cat
            elif op == "<=" and t >= 0.5:
                by_var[var]["exclude"].add(cat)
    parts = []
    for var, st in by_var.items():
        cats_all = var_to_cats[var]["all"]
        if st["forced"] is not None:
            parts.append(f"{var} = {st['forced']}")
        elif len(st["exclude"]) > 0:
            allowed = [c for c in cats_all if c not in st["exclude"]]
            if len(allowed) <= max(1, int(0.6*len(cats_all))):
                parts.append(f"{var} ∈ {{{', '.join(allowed)}}}")
            else:
                parts.append(f"{var} ∉ {{{', '.join(sorted(st['exclude']))}}}")
    return "; ".join(parts) if parts else "(no restrictions)"

var_to_cats, dummy_to_pair = build_ohe_maps(prep, FEATURES, X.columns.tolist())
leaf_paths = tree_leaf_paths(tree, X.columns.tolist())
stats_lookup = leaf_tbl.set_index("leaf")[["n","mean_obj"]].to_dict(orient="index")

rows = []
for lf in leaf_tbl_top["leaf"].tolist():
    conds = leaf_paths.get(lf, [])
    rule = conditions_to_rule(var_to_cats, dummy_to_pair, conds)
    st = stats_lookup.get(lf, {"n": np.nan, "mean_obj": np.nan})
    rows.append({"leaf": int(lf), "n_honest": int(st["n"]), "mean_objective": float(st["mean_obj"]), "rule": rule})
leaf_rules = pd.DataFrame(rows).sort_values("mean_objective", ascending=False)
leaf_rules_path = os.path.join(BASE_DIR, f"leaf_rules__{OBJECTIVE_ARR}.csv")
leaf_rules.to_csv(leaf_rules_path, index=False)
print(f"[Rules] Per-leaf rules -> {leaf_rules_path}")

for g in selected_groups:
    if g["leaves"] is None or len(g["leaves"]) == 0:
        continue
    lines = [f"# Subgroup {g['tag']}",
             f"n = {g['n']}  |  mean objective = {g['mean_obj']:.4f}",
             "", "This subgroup is the union (OR) of the following leaf rules:"]
    for lf in sorted(list(g["leaves"])):
        conds = leaf_paths.get(lf, [])
        rule = conditions_to_rule(var_to_cats, dummy_to_pair, conds)
        st = stats_lookup.get(lf, {"n": np.nan, "mean_obj": np.nan})
        lines.append(f"- **Leaf {lf}**: {rule}  (n_honest={st['n']}, mean={float(st['mean_obj']):.4f})")
    out_md = os.path.join(BASE_DIR, f"ssr_rulebook__{OBJECTIVE_ARR}__{g['tag']}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Rules] Rulebook for {g['tag']} -> {out_md}")

# =======================
# Microsimulation helpers
# =======================
def _fit_best(dur, evt):
    cands = []
    for nm, F in [("Weibull", WeibullFitter),
                  ("LogLogistic", LogLogisticFitter),
                  ("LogNormal", LogNormalFitter)]:
        try:
            m = F().fit(dur, event_observed=evt)
            aic = getattr(m, "AIC_", float("inf"))
            cands.append((aic, nm, m))
        except Exception:
            pass
    if not cands:
        raise RuntimeError("Parametric survival fit failed (no converged model).")
    cands.sort(key=lambda z: z[0])
    return cands[0][2]

# Precompute flags for S-only fitting
any_surgery   = to_bool(prep.get("any_surgery", pd.Series(False, index=prep.index)))
any_radiation = to_bool(prep.get("any_radiation", pd.Series(False, index=prep.index)))
surv_days_dx  = to_num(prep.get("surv_days_from_dx"))
valid_120     = (surv_days_dx - 120.0 >= 0) | (surv_days_dx - 120.0).isna()

def fit_surv_S_arm(sub_ids_for_fit=None):
    """
    Fit parametric survival (from 120d) on the S-only arm for:
      - BCSM
      - Other-cause death
      - Progression to invasive cancer
    If subgroup S-only sample is small (< MIN_S_BASE_N), fit on overall S-only.
    If progression timing isn't available, fall back to a flat tiny hazard.
    """
    mask_Sonly = (any_surgery & ~any_radiation) & valid_120.fillna(False)

    if sub_ids_for_fit is not None:
        mask_sub = mask_Sonly & prep[IDCOL].astype(str).isin([str(x) for x in sub_ids_for_fit])
        fit_mask = mask_sub if mask_sub.sum() >= MIN_S_BASE_N else mask_Sonly
    else:
        fit_mask = mask_Sonly

    base = prep[fit_mask].copy()
    if base.empty:
        raise RuntimeError("No S-only patients available for baseline fit.")

    # --- BCSM & Other (from 120d)
    t_land  = to_num(base.get("surv_days_from_dx")) - 120.0
    e_bcsm  = to_bool(base.get("event_bcsm", pd.Series(False, index=base.index))).astype(int)
    e_other = (to_bool(base.get("event_any",  pd.Series(False, index=base.index))) &
               (~to_bool(base.get("event_bcsm", pd.Series(False, index=base.index))))).astype(int)

    ok_b = np.isfinite(t_land) & (t_land >= 0)
    t_b = t_land[ok_b].to_numpy()
    e_b = e_bcsm[ok_b].to_numpy()
    e_o = e_other[ok_b].to_numpy()
    if t_b.size == 0:
        raise RuntimeError("All durations invalid for BCSM/Other baseline fit.")
    m_bcsm_S  = _fit_best(t_b, e_b)
    m_other_S = _fit_best(t_b, e_o)

    # --- Progression (robust)
    # Prefer 't_prog_from_120d_days'; else derive; else NaN -> fallback
    if "t_prog_from_120d_days" in base.columns:
        t_prog = pd.to_numeric(base["t_prog_from_120d_days"], errors="coerce")
    elif "prog_days_from_dx" in base.columns:
        t_prog = pd.to_numeric(base["prog_days_from_dx"], errors="coerce") - 120.0
    else:
        t_prog = pd.Series(np.nan, index=base.index, dtype=float)

    if "event_prog" in base.columns:
        e_prog = pd.to_numeric(base["event_prog"], errors="coerce").fillna(0).astype(int)
    else:
        e_prog = pd.Series(0, index=base.index, dtype=int)

    ok_p = np.isfinite(t_prog) & (t_prog >= 0)

    if ok_p.sum() > 0:
        t_p = t_prog[ok_p].to_numpy()
        e_p = e_prog[ok_p].to_numpy()
        try:
            m_prog_S = _fit_best(t_p, e_p)
        except Exception:
            class _FlatProg:
                def survival_function_at_times(self, times):
                    lam = 0.001
                    S = np.exp(-lam * (times / DT_DAYS))
                    return pd.Series(S, index=times)
            m_prog_S = _FlatProg()
            warnings.warn("[Progression] Parametric fit failed; using flat tiny hazard baseline.")
    else:
        class _FlatProg:
            def survival_function_at_times(self, times):
                lam = 0.001
                S = np.exp(-lam * (times / DT_DAYS))
                return pd.Series(S, index=times)
        m_prog_S = _FlatProg()
        warnings.warn("[Progression] No valid progression times; using flat tiny hazard baseline.")

    return m_bcsm_S, m_other_S, m_prog_S

def survival_series(model, years, dt_days=DT_DAYS):
    months = int(np.ceil(years*12))
    times = np.arange(0, months+1) * dt_days
    S = model.survival_function_at_times(times)
    S_vals = np.asarray(getattr(S, "values", S)).ravel()
    S_vals = np.maximum(1e-12, S_vals)
    out = pd.Series(S_vals, index=np.arange(len(times)))
    out.index.name = "month"
    return out

def hazard_from_survival_curve(S_arr):
    # S_arr: numpy array of survival values per month index
    S_prev = S_arr[:-1]
    S_next = S_arr[1:]
    h = (1.0 - (S_next / S_prev))
    return np.clip(h, 0.0, 1.0)

# Econ inputs
u_alive_base = float(UTILS["U_ALIVE"])
du_prog      = float(UTILS["DU_PROG_CHRONIC"])
du_s         = float(UTILS["DU_S_ACUTE"]);   s_mths  = int(UTILS["S_ACUTE_MONTHS"])
du_rt        = float(UTILS["DU_RT_ACUTE"]);  rt_mths = int(UTILS["RT_ACUTE_MONTHS"])

C_OBS_INIT = float(COSTS["C_OBS_INIT"])
C_S_INIT   = float(COSTS["C_S_INIT"])
C_RT       = float(COSTS["C_RT_COURSE"])
C_FU       = float(COSTS["C_MONTHLY_FOLLOWUP"])
C_PROG_I   = float(COSTS["C_PROG_INIT"])
C_PROG_M   = float(COSTS["C_PROG_MONTHLY"])
C_EOL_B    = float(COSTS["C_EOL_BCSM"])
C_EOL_O    = float(COSTS["C_EOL_OTHER"])

def simulate_three_arm_with_prog(
    S_bcsm_S, S_other_S, S_prog_S,
    theta_bcsm_S_vs_OBS, theta_bcsm_SRT_vs_S,
    theta_prog_S_vs_OBS, theta_prog_SRT_vs_S,
    seed_base
):
    """
    CRN across arms: two random streams per month (death, progression).
    Other-cause hazards unaffected by treatment across arms.
    """
    months = len(S_bcsm_S) - 1
    # Baseline (S arm) hazards from survival
    q_bcsm_S  = hazard_from_survival_curve(S_bcsm_S.values)
    q_other   = hazard_from_survival_curve(S_other_S.values)
    q_prog_S  = hazard_from_survival_curve(S_prog_S.values)

    # PH transforms
    thB_obs  = np.clip(theta_bcsm_S_vs_OBS,  0.05, 3.0)
    thB_srt  = np.clip(theta_bcsm_SRT_vs_S,  0.05, 3.0)
    thP_obs  = np.clip(theta_prog_S_vs_OBS,  0.05, 3.0)
    thP_srt  = np.clip(theta_prog_SRT_vs_S,  0.05, 3.0)

    S_bcsm_OBS = np.power(S_bcsm_S.values, 1.0 / thB_obs)
    S_bcsm_SRT = np.power(S_bcsm_S.values, thB_srt)
    q_bcsm_OBS = hazard_from_survival_curve(S_bcsm_OBS)
    q_bcsm_SRT = hazard_from_survival_curve(S_bcsm_SRT)

    S_prog_OBS = np.power(S_prog_S.values, 1.0 / thP_obs)
    S_prog_SRT = np.power(S_prog_S.values, thP_srt)
    q_prog_OBS = hazard_from_survival_curve(S_prog_OBS)
    q_prog_SRT = hazard_from_survival_curve(S_prog_SRT)

    dfact = (1.0 / (1.0 + DISCOUNT_RATE_ANNUAL)) ** (np.arange(months)/12.0)
    out = {}
    rng = np.random.RandomState(seed_base)
    r_death = rng.rand(months)
    r_prog  = rng.rand(months)

    for arm, q_b, q_p in [
        ("OBS", q_bcsm_OBS, q_prog_OBS),
        ("S",   q_bcsm_S,   q_prog_S),
        ("S+RT",q_bcsm_SRT, q_prog_SRT)
    ]:
        alive = True; progressed = False; prog_m = None
        cause = "alive"; death_m = months
        cost = (C_OBS_INIT if arm=="OBS" else C_S_INIT + (C_RT if arm=="S+RT" else 0.0))
        qaly = 0.0

        for m in range(months):
            if not alive:
                break

            # Independent monthly hazards (approximation):
            h_b = q_b[m]
            h_o = q_other[m]
            h_p = (0.0 if progressed else q_p[m])

            # Competing deaths first
            p_b = h_b * (1.0 - h_o)
            p_o = (1.0 - h_b) * h_o
            r1 = r_death[m]
            if r1 < p_b:
                alive=False; cause="bcsm"; death_m=m
            elif r1 < p_b + p_o:
                alive=False; cause="other"; death_m=m
            else:
                # Survived this month -> check progression (if not already)
                if (not progressed) and (r_prog[m] < h_p):
                    progressed = True; prog_m = m
                    cost += C_PROG_I * dfact[m]

                # Utility (chronic disutility after progression)
                util = max(0.0, u_alive_base - (du_prog if progressed else 0.0))
                # Acute disutilities
                if arm in ("S","S+RT") and m < s_mths: util = max(0.0, util - du_s)
                if arm=="S+RT" and m < rt_mths:        util = max(0.0, util - du_rt)

                qaly += util * (dfact[m] * (DT_DAYS/365.25))
                # Monthly costs
                cost += (C_FU + (C_PROG_M if progressed else 0.0)) * dfact[m]

        # EOL cost
        if cause=="bcsm":
            cost += C_EOL_B * (dfact[min(death_m, months-1)] if death_m < months else 1.0)
        elif cause=="other":
            cost += C_EOL_O * (dfact[min(death_m, months-1)] if death_m < months else 1.0)

        out[arm] = dict(cost=cost, qaly=qaly, death_m=death_m, cause=cause,
                        progressed=int(progressed), prog_m=(prog_m if progressed else -1))
    return out

def icer_pairwise(df_pt):
    rows = []
    def add(a, b):
        dC = df_pt[f"cost_{a}"].mean() - df_pt[f"cost_{b}"].mean()
        dQ = df_pt[f"qaly_{a}"].mean() - df_pt[f"qaly_{b}"].mean()
        status = ""
        ICER = np.nan
        if dQ <= 0 and dC >= 0:
            status = "Dominated"
        elif dQ > 0:
            ICER = dC / dQ
        rows.append(dict(comparison=f"{a} vs {b}", dC=dC, dQ=dQ, ICER=ICER, status=status))
    add("S", "OBS")
    add("SRT", "S")
    add("SRT", "OBS")
    return pd.DataFrame(rows)

def icer_frontier(df_pt):
    """
    Return all strategies with dominance status.
    - 'dom_status' ∈ {On frontier, Extendedly dominated, Strongly dominated}
    - Incremental columns (inc_cost, inc_qaly, ICER) are filled only for the frontier.
    """
    import numpy as np
    import pandas as pd

    # Mean cost/QALY for each arm
    means = pd.DataFrame({
        "strategy": ["OBS", "S", "SRT"],
        "mean_cost": [
            df_pt["cost_OBS"].mean(),
            df_pt["cost_S"].mean(),
            df_pt["cost_SRT"].mean()
        ],
        "mean_qaly": [
            df_pt["qaly_OBS"].mean(),
            df_pt["qaly_S"].mean(),
            df_pt["qaly_SRT"].mean()
        ],
    }).sort_values(["mean_cost", "mean_qaly"]).reset_index(drop=True)

    n = len(means)

    # -------- Strong dominance (keep all, just mark)
    strong_dom = set()
    for i in range(n):
        ci, qi = means.loc[i, "mean_cost"], means.loc[i, "mean_qaly"]
        for j in range(n):
            if i == j:
                continue
            cj, qj = means.loc[j, "mean_cost"], means.loc[j, "mean_qaly"]
            # strong dominance: >= cost and <= QALY, with at least one strict
            if (ci >= cj and qi <= qj) and (ci > cj or qi < qj):
                strong_dom.add(i)
                break

    # -------- Extended dominance on the set with no strong dominance
    # Work on indices sorted by cost
    eff_idx = [i for i in range(n) if i not in strong_dom]
    eff_idx = sorted(eff_idx, key=lambda k: (means.loc[k, "mean_cost"], means.loc[k, "mean_qaly"]))

    changed = True
    while changed and len(eff_idx) >= 3:
        changed = False
        # check convexity via adjacent ICER slopes
        for k in range(1, len(eff_idx) - 1):
            i_prev, i_curr, i_next = eff_idx[k-1], eff_idx[k], eff_idx[k+1]
            dC1 = means.loc[i_curr, "mean_cost"] - means.loc[i_prev, "mean_cost"]
            dQ1 = means.loc[i_curr, "mean_qaly"] - means.loc[i_prev, "mean_qaly"]
            dC2 = means.loc[i_next, "mean_cost"] - means.loc[i_curr, "mean_cost"]
            dQ2 = means.loc[i_next, "mean_qaly"] - means.loc[i_curr, "mean_qaly"]

            icer1 = np.inf if dQ1 <= 0 else dC1 / dQ1
            icer2 = np.inf if dQ2 <= 0 else dC2 / dQ2

            # Extended dominance: earlier slope ≥ next slope → remove middle
            if icer1 >= icer2 - 1e-12:
                eff_idx.pop(k)
                changed = True
                break

    frontier_idx = eff_idx  # indices (in 'means') that are on the efficient frontier

    # -------- Build output with dominance status
    dom_status = []
    frontier_rank = [np.nan] * n
    for i in range(n):
        if i in strong_dom:
            dom_status.append("Strongly dominated")
        elif i in frontier_idx:
            dom_status.append("On frontier")
        else:
            dom_status.append("Extendedly dominated")

    # Incremental metrics only for frontier rows (ordered by cost)
    inc_cost = [np.nan] * n
    inc_qaly = [np.nan] * n
    ICER     = [np.nan] * n

    for pos, idx in enumerate(frontier_idx):
        frontier_rank[idx] = pos + 1
        if pos == 0:
            continue
        prev = frontier_idx[pos - 1]
        dC = means.loc[idx, "mean_cost"] - means.loc[prev, "mean_cost"]
        dQ = means.loc[idx, "mean_qaly"] - means.loc[prev, "mean_qaly"]
        inc_cost[idx] = dC
        inc_qaly[idx] = dQ
        ICER[idx] = (dC / dQ) if dQ > 0 else np.nan

    out = means.copy()
    out["inc_cost"] = inc_cost
    out["inc_qaly"] = inc_qaly
    out["ICER"] = ICER
    out["dom_status"] = dom_status
    out["frontier_rank"] = frontier_rank

    # Optional: sort by cost for readability
    out = out.sort_values(["mean_cost", "mean_qaly"]).reset_index(drop=True)
    return out


# =======================
# Run overall + subgroup CEA
# =======================
for g in selected_groups:
    TAG = g["tag"]
    members = set(map(str, g["members"]))
    M = DF[DF[IDCOL].isin(members)].copy()
    if M.empty:
        print(f"[Microsim] {TAG}: no members, skip.")
        continue

    # Fit baseline S-arm survival for subgroup (fallback automatic inside)
    m_bcsm_S, m_other_S, m_prog_S = fit_surv_S_arm(sub_ids_for_fit=members)
    S_bcsm_S  = survival_series(m_bcsm_S, MAX_YEARS, DT_DAYS)
    S_other_S = survival_series(m_other_S, MAX_YEARS, DT_DAYS)
    S_prog_S  = survival_series(m_prog_S, MAX_YEARS, DT_DAYS)

    # Simulate per patient (3 arms)
    rows = []
    cols_needed = [IDCOL, "theta_bcsm_S_vs_OBS","theta_bcsm_SRT_vs_S","theta_prog_S_vs_OBS","theta_prog_SRT_vs_S"]
    for k, r in enumerate(M[cols_needed].itertuples(index=False)):
        pid, thB_obs, thB_srt, thP_obs, thP_srt = r
        sim = simulate_three_arm_with_prog(
            S_bcsm_S, S_other_S, S_prog_S,
            float(thB_obs), float(thB_srt),
            float(thP_obs), float(thP_srt),
            seed_base=100000 + k
        )
        rows.append({
            IDCOL: pid,
            "cost_OBS": sim["OBS"]["cost"],  "qaly_OBS": sim["OBS"]["qaly"],
            "cost_S":   sim["S"]["cost"],    "qaly_S":   sim["S"]["qaly"],
            "cost_SRT": sim["S+RT"]["cost"], "qaly_SRT": sim["S+RT"]["qaly"],
            "death_m_OBS": sim["OBS"]["death_m"], "cause_OBS": sim["OBS"]["cause"],
            "death_m_S":   sim["S"]["death_m"],   "cause_S":   sim["S"]["cause"],
            "death_m_SRT": sim["S+RT"]["death_m"],"cause_SRT": sim["S+RT"]["cause"],
            "progressed_OBS": sim["OBS"]["progressed"], "prog_m_OBS": sim["OBS"]["prog_m"],
            "progressed_S":   sim["S"]["progressed"],   "prog_m_S":   sim["S"]["prog_m"],
            "progressed_SRT": sim["S+RT"]["progressed"],"prog_m_SRT": sim["S+RT"]["prog_m"]
        })
    res = pd.DataFrame(rows)

    # Save patient-level
    OUT_PT = os.path.join(BASE_DIR, f"msim3prog_patient_level__{TAG}.csv")
    res.to_csv(OUT_PT, index=False)

    # Summary with NMB CE-probs at multiple WTP
    def summarize(df, lambdas=(50000,100000,150000,200000)):
        out=[]
        mean_cost = dict(OBS=df["cost_OBS"].mean(), S=df["cost_S"].mean(), SRT=df["cost_SRT"].mean())
        mean_qaly = dict(OBS=df["qaly_OBS"].mean(), S=df["qaly_S"].mean(), SRT=df["qaly_SRT"].mean())
        for lam in lambdas:
            nmb_OBS = lam*df["qaly_OBS"] - df["cost_OBS"]
            nmb_S   = lam*df["qaly_S"]   - df["cost_S"]
            nmb_SRT = lam*df["qaly_SRT"] - df["cost_SRT"]
            best = np.argmax(np.vstack([nmb_OBS, nmb_S, nmb_SRT]), axis=0)  # 0 OBS,1 S,2 SRT
            out.append(dict(
                WTP=lam,
                mean_cost_OBS=mean_cost["OBS"], mean_qaly_OBS=mean_qaly["OBS"],
                mean_cost_S=mean_cost["S"],     mean_qaly_S=mean_qaly["S"],
                mean_cost_SRT=mean_cost["SRT"], mean_qaly_SRT=mean_qaly["SRT"],
                CE_prob_OBS=(best==0).mean(), CE_prob_S=(best==1).mean(), CE_prob_SRT=(best==2).mean()
            ))
        return pd.DataFrame(out)

    summ = summarize(res)
    OUT_SUM = os.path.join(BASE_DIR, f"msim3prog_summary__{TAG}.csv")
    summ.to_csv(OUT_SUM, index=False)

    # ICERs
    pair = icer_pairwise(res)
    OUT_PAIR = os.path.join(BASE_DIR, f"icer_table_pairwise__{TAG}.csv")
    pair.to_csv(OUT_PAIR, index=False)

    front = icer_frontier(res)
    OUT_FRONT = os.path.join(BASE_DIR, f"icer_table_frontier__{TAG}.csv")
    front.to_csv(OUT_FRONT, index=False)

    print(f"[Microsim:{TAG}] Saved:\n  {OUT_PT}\n  {OUT_SUM}\n  {OUT_PAIR}\n  {OUT_FRONT}")

print("STEP-04 (policy + SSR + 3-arm CEA with progression) completed.")
