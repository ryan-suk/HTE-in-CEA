# =======================
# STEP-04 (fixed): Subgroup 3-arm CEA (OBS, S, S+RT) with HTE
# - Robust ID handling (patient_id vs Patient ID) via id_std
# - Always runs "AllEligible"
# - Writes a catalog of category labels
# - Auto-runs by level for key variables
# =======================
import os, re, json, warnings
import numpy as np
import pandas as pd
from lifelines import WeibullFitter, LogLogisticFitter, LogNormalFitter
warnings.filterwarnings("ignore")

# -------- Paths --------
BASE_DIR  = r"C:\Users\ryann\OneDrive\A_2025-26_Manuscripts\DCIS HTE\SEER files"
PATH_PREP = os.path.join(BASE_DIR, "dcis_prepared.csv")
PATH_HR_S_OBS  = os.path.join(BASE_DIR, "hr_s_vs_obs_by_patient.csv")
PATH_HR_SRT_S  = os.path.join(BASE_DIR, "hr_splusrt_vs_s_by_patient.csv")

# -------- Core toggles --------
DISCOUNT_RATE_ANNUAL = 0.03
DT_DAYS   = 30.4375
MAX_YEARS = 40
WTP_LIST  = (50000, 100000, 150000, 200000)

# Subgroup execution controls
MIN_N_SUB     = 50       # run if n >= this value
ALLOW_SMALL   = True     # if True, run even when n < MIN_N_SUB (warn)
AUTO_VARS     = ["age_recode","grade_dcisset","tumor_size_recode","race_origin_recode"]

# -------- Placeholders (replace with your values) --------
COSTS = dict(
    C_OBS_INIT           = 0.0,
    C_S_INIT             = 8000.0,
    C_RT_COURSE          = 8000.0,
    C_MONTHLY_FOLLOWUP   = 50.0,
    C_EOL_BCSM           = 12000.0,
    C_EOL_OTHER          = 8000.0
)
UTILS = dict(
    U_ALIVE         = 0.83,
    DU_S_ACUTE      = 0.01,
    S_ACUTE_MONTHS  = 1,
    DU_RT_ACUTE     = 0.02,
    RT_ACUTE_MONTHS = 2
)

# ---------- helpers ----------
def to_bool(s):
    """Robust boolean parse from Series/array/scalar to a boolean Series.
    Truthy tokens: true, 1, t, y, yes (case/space-insensitive).
    Everything else (false, 0, no, nan, empty) → False.
    """
    import pandas as pd
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    tokens = s.astype(str).str.strip().str.lower()
    return tokens.isin(["true", "1", "t", "y", "yes"])
def to_num(s):  return pd.to_numeric(s, errors="coerce")

def standardize_id_col(df_like):
    """Return frame with an added 'id_std' column derived from patient_id or Patient ID."""
    df2 = df_like.copy()
    if "patient_id" in df2.columns:
        df2["id_std"] = df2["patient_id"].astype(str)
        return df2, "patient_id"
    if "Patient ID" in df2.columns:
        df2["id_std"] = df2["Patient ID"].astype(str)
        return df2, "Patient ID"
    raise ValueError("No patient ID column found (need 'patient_id' or 'Patient ID').")

# ---------- load prepared ----------
prep = pd.read_csv(PATH_PREP, dtype=str).rename(columns=str.strip)
prep, IDCOL_ORIG = standardize_id_col(prep)

# ---------- load HR tables (θ for S vs OBS, and S+RT vs S) ----------
hr_s_obs = pd.read_csv(PATH_HR_S_OBS, dtype=str).rename(columns=str.strip)
hr_srt_s = pd.read_csv(PATH_HR_SRT_S, dtype=str).rename(columns=str.strip)

hr_s_obs, IDCOL_HR1 = standardize_id_col(hr_s_obs)
hr_srt_s, IDCOL_HR2 = standardize_id_col(hr_srt_s)

hr_s_obs["theta_S_vs_OBS"] = pd.to_numeric(hr_s_obs["theta_use"], errors="coerce")
hr_srt_s["theta_SRT_vs_S"] = pd.to_numeric(hr_srt_s["theta_use"], errors="coerce")

# Merge by id_std; keep original ID column for outputs
DF = prep.merge(hr_s_obs[["id_std","theta_S_vs_OBS"]], on="id_std", how="inner") \
         .merge(hr_srt_s[["id_std","theta_SRT_vs_S"]], on="id_std", how="inner")
DF["theta_SRT_vs_OBS"] = DF["theta_S_vs_OBS"] * DF["theta_SRT_vs_S"]

if DF.empty:
    raise RuntimeError(
        "After merging θ files with prepared data, no rows remained. "
        "Check that HR CSVs and dcis_prepared.csv share the same IDs."
    )

# ---------- baseline S-arm survival fit (from 120d) ----------
any_surgery   = to_bool(DF.get("any_surgery", pd.Series(False, index=DF.index)))
any_radiation = to_bool(DF.get("any_radiation", pd.Series(False, index=DF.index)))
surv_days_dx  = to_num(DF.get("surv_days_from_dx"))
t_from120     = surv_days_dx - 120.0
valid_120     = (t_from120 >= 0) | t_from120.isna()

S_mask = (any_surgery & ~any_radiation) & valid_120.fillna(False)
Base = DF[S_mask].copy()
if Base.empty:
    raise RuntimeError("No Surgery-only patients to fit baseline on.")

tS = to_num(Base["surv_days_from_dx"]) - 120.0
e_bcsm  = to_bool(Base.get("event_bcsm", pd.Series(False, index=Base.index))).astype(int)
e_other = (to_bool(Base.get("event_any", pd.Series(False, index=Base.index))) &
          (~to_bool(Base.get("event_bcsm", pd.Series(False, index=Base.index))))).astype(int)

def fit_best(durations, events):
    cands=[]
    for name, F in [("Weibull", WeibullFitter), ("LogLogistic", LogLogisticFitter), ("LogNormal", LogNormalFitter)]:
        try:
            m = F().fit(durations, event_observed=events)
            aic = getattr(m, "AIC_", float("inf"))
            cands.append((aic, name, m))
        except Exception:
            pass
    if not cands: raise RuntimeError("All parametric fits failed.")
    cands.sort(key=lambda z: z[0])
    return cands[0]  # (AIC, name, model)

(best_bcsm_AIC, best_bcsm_name, M_BCSM)   = fit_best(tS, e_bcsm)
(best_other_AIC, best_other_name, M_OTHER)= fit_best(tS, e_other)

def survival_series(model, years, dt_days=DT_DAYS):
    months = int(np.ceil(years*12))
    times = np.arange(0, months+1) * dt_days
    S = model.survival_function_at_times(times)
    S = pd.Series(np.maximum(1e-12, S.values.ravel()), index=np.arange(len(times)))
    return S

def discrete_hazard_from_S(S):
    S1 = S.shift(-1)
    q = 1.0 - (S1/S)
    return q.iloc[:-1].clip(0.0, 1.0)

S_bcsm_S  = survival_series(M_BCSM, MAX_YEARS, DT_DAYS)
S_other_S = survival_series(M_OTHER, MAX_YEARS, DT_DAYS)
q_bcsm_S  = discrete_hazard_from_S(S_bcsm_S).values
q_other   = discrete_hazard_from_S(S_other_S).values
months    = len(q_bcsm_S)

# ---------- microsim machinery ----------
u_alive    = float(UTILS["U_ALIVE"])
du_s       = float(UTILS["DU_S_ACUTE"]);   s_mths  = int(UTILS["S_ACUTE_MONTHS"])
du_rt      = float(UTILS["DU_RT_ACUTE"]);  rt_mths = int(UTILS["RT_ACUTE_MONTHS"])
C_OBS_INIT = float(COSTS["C_OBS_INIT"])
C_S_INIT   = float(COSTS["C_S_INIT"])
C_RT       = float(COSTS["C_RT_COURSE"])
C_FU       = float(COSTS["C_MONTHLY_FOLLOWUP"])
C_EOL_B    = float(COSTS["C_EOL_BCSM"])
C_EOL_O    = float(COSTS["C_EOL_OTHER"])
dfact = (1.0 / (1.0 + DISCOUNT_RATE_ANNUAL)) ** (np.arange(months)/12.0)

def hazard_from_S_power(S, theta):
    S_arm = np.power(np.clip(S, 1e-12, 1.0), float(theta))
    return (1.0 - (S_arm[1:]/S_arm[:-1])).clip(0, 1)

def simulate_three_arm(theta_S_vs_OBS, theta_SRT_vs_S, base_seed):
    q_b_obs = hazard_from_S_power(S_bcsm_S.values, 1.0/float(theta_S_vs_OBS))
    q_b_s   = q_bcsm_S
    q_b_srt = hazard_from_S_power(S_bcsm_S.values, float(theta_SRT_vs_S))
    q_o     = q_other

    out={}
    rng = np.random.RandomState(base_seed)
    for arm, q_b in [("OBS", q_b_obs), ("S", q_b_s), ("S+RT", q_b_srt)]:
        alive=True; cause="alive"; death_m=months
        cost = (C_OBS_INIT if arm=="OBS" else (C_S_INIT + (C_RT if arm=="S+RT" else 0.0)))
        qaly = 0.0
        for m in range(months):
            if not alive: break
            p_b = q_b[m] * (1.0 - q_o[m])
            p_o = (1.0 - q_b[m]) * q_o[m]
            r = rng.rand()
            if r < p_b:
                alive=False; cause="bcsm"; death_m=m
            elif r < p_b + p_o:
                alive=False; cause="other"; death_m=m
            else:
                util = u_alive
                if arm in ("S","S+RT") and m < s_mths: util = max(0.0, util - du_s)
                if arm=="S+RT" and m < rt_mths:       util = max(0.0, util - du_rt)
                qaly += util * (dfact[m] * (DT_DAYS/365.25))
                cost += C_FU * dfact[m]
        dfm = dfact[min(death_m, months-1)] if death_m < months else 1.0
        if cause=="bcsm":  cost += C_EOL_B * dfm
        elif cause=="other": cost += C_EOL_O * dfm
        out[arm] = dict(cost=cost, qaly=qaly, death_m=death_m, cause=cause)
    return out

def summarize_three(res_df, lambdas=WTP_LIST):
    out=[]
    mean_cost = dict(OBS=res_df["cost_OBS"].mean(), S=res_df["cost_S"].mean(), SRT=res_df["cost_SRT"].mean())
    mean_qaly = dict(OBS=res_df["qaly_OBS"].mean(), S=res_df["qaly_S"].mean(), SRT=res_df["qaly_SRT"].mean())
    inc_S_vs_OBS  = dict(dC=res_df["dC_S_vs_OBS"].mean(),  dQ=res_df["dQ_S_vs_OBS"].mean())
    inc_SRT_vs_S  = dict(dC=res_df["dC_SRT_vs_S"].mean(),  dQ=res_df["dQ_SRT_vs_S"].mean())
    inc_SRT_vs_OBS= dict(dC=res_df["dC_SRT_vs_OBS"].mean(),dQ=res_df["dQ_SRT_vs_OBS"].mean())
    for lam in lambdas:
        nmb_OBS = lam*res_df["qaly_OBS"] - res_df["cost_OBS"]
        nmb_S   = lam*res_df["qaly_S"]   - res_df["cost_S"]
        nmb_SRT = lam*res_df["qaly_SRT"] - res_df["cost_SRT"]
        best = np.argmax(np.vstack([nmb_OBS, nmb_S, nmb_SRT]), axis=0)
        out.append(dict(
            WTP=lam,
            mean_cost_OBS=mean_cost["OBS"], mean_qaly_OBS=mean_qaly["OBS"],
            mean_cost_S=mean_cost["S"],     mean_qaly_S=mean_qaly["S"],
            mean_cost_SRT=mean_cost["SRT"], mean_qaly_SRT=mean_qaly["SRT"],
            inc_S_vs_OBS_dC=inc_S_vs_OBS["dC"], inc_S_vs_OBS_dQ=inc_S_vs_OBS["dQ"],
            inc_SRT_vs_S_dC=inc_SRT_vs_S["dC"], inc_SRT_vs_S_dQ=inc_SRT_vs_S["dQ"],
            inc_SRT_vs_OBS_dC=inc_SRT_vs_OBS["dC"], inc_SRT_vs_OBS_dQ=inc_SRT_vs_OBS["dQ"],
            CE_prob_OBS=(best==0).mean(), CE_prob_S=(best==1).mean(), CE_prob_SRT=(best==2).mean()
        ))
    return pd.DataFrame(out)

def write_icer_tables(res_df, prefix):
    m_cost = dict(OBS=res_df["cost_OBS"].mean(), S=res_df["cost_S"].mean(), SRT=res_df["cost_SRT"].mean())
    m_qaly = dict(OBS=res_df["qaly_OBS"].mean(), S=res_df["qaly_S"].mean(), SRT=res_df["qaly_SRT"].mean())

    def pair_icer(high, low):
        dC = m_cost[high] - m_cost[low]
        dQ = m_qaly[high] - m_qaly[low]
        icer = (dC/dQ) if dQ>0 else np.nan
        status = ""
        if dQ <= 0 and dC >= 0: status = "Dominated"
        elif dQ > 0 and dC < 0: status = "Dominant"
        return dict(comparison=f"{high} vs {low}", dC=dC, dQ=dQ, ICER=icer, status=status)

    pair_df = pd.DataFrame([
        pair_icer("S", "OBS"),
        pair_icer("SRT", "S"),
        pair_icer("SRT", "OBS"),
    ])
    pair_path = os.path.join(BASE_DIR, f"icer_table_pairwise_{prefix}.csv")
    pair_df.to_csv(pair_path, index=False)

    front = pd.DataFrame([
        {"strategy":"OBS","mean_cost":m_cost["OBS"],"mean_qaly":m_qaly["OBS"]},
        {"strategy":"S",  "mean_cost":m_cost["S"],  "mean_qaly":m_qaly["S"]},
        {"strategy":"SRT","mean_cost":m_cost["SRT"],"mean_qaly":m_qaly["SRT"]},
    ])

    def strictly_dominated(df):
        dom_idx=set()
        for i, ri in df.iterrows():
            for j, rj in df.iterrows():
                if i==j: continue
                if (rj["mean_cost"]<=ri["mean_cost"]) and (rj["mean_qaly"]>=ri["mean_qaly"]) and \
                   ((rj["mean_cost"]<ri["mean_cost"]) or (rj["mean_qaly"]>ri["mean_qaly"])):
                    dom_idx.add(i); break
        return df.drop(index=list(dom_idx)).copy()

    def incr_table(df_sorted):
        rows=[]
        for k in range(len(df_sorted)):
            s, mc, mq = df_sorted.loc[k, ["strategy","mean_cost","mean_qaly"]]
            if k==0:
                rows.append(dict(strategy=s, comparator="", inc_cost=np.nan, inc_qaly=np.nan, ICER=np.nan,
                                 mean_cost=mc, mean_qaly=mq))
            else:
                dC = mc - df_sorted.loc[k-1, "mean_cost"]
                dQ = mq - df_sorted.loc[k-1, "mean_qaly"]
                icer = (dC/dQ) if dQ>0 else np.nan
                rows.append(dict(strategy=s, comparator=df_sorted.loc[k-1,"strategy"],
                                 inc_cost=dC, inc_qaly=dQ, ICER=icer,
                                 mean_cost=mc, mean_qaly=mq))
        return pd.DataFrame(rows)

    def remove_extended_dominance(df):
        df = df.sort_values("mean_cost").reset_index(drop=True)
        changed=True
        while changed and len(df)>=3:
            changed=False
            tab=incr_table(df)
            for k in range(1,len(tab)-1):
                dC = df.loc[k+1,"mean_cost"] - df.loc[k,"mean_cost"]
                dQ = df.loc[k+1,"mean_qaly"] - df.loc[k,"mean_qaly"]
                icer_next = (dC/dQ) if dQ>0 else np.inf
                if np.isfinite(tab.loc[k,"ICER"]) and np.isfinite(icer_next) and (tab.loc[k,"ICER"]>icer_next):
                    df = df.drop(index=k).reset_index(drop=True)
                    changed=True; break
        return df

    front_nd = strictly_dominated(front)
    front_nd = remove_extended_dominance(front_nd).sort_values("mean_cost").reset_index(drop=True)
    frontier = incr_table(front_nd)

    front_path = os.path.join(BASE_DIR, f"icer_table_frontier_{prefix}.csv")
    frontier.to_csv(front_path, index=False)

    by = front.sort_values("mean_cost").reset_index(drop=True)
    by_path = os.path.join(BASE_DIR, f"icer_by_strategy_{prefix}.csv")
    by.to_csv(by_path, index=False)

    return pair_path, front_path, by_path

# ---------- subgroup utilities ----------
def subgroup_mask(df0, criteria):
    """criteria = {col: {'contains_any':[...]} or {'equals_any':[...]} or {'range':[lo,hi]} or scalar equality}"""
    if not criteria:  # no criteria -> everyone
        return pd.Series(True, index=df0.index)
    mask = pd.Series(True, index=df0.index)
    for col, rule in criteria.items():
        if col not in df0.columns:
            print(f"[Warn] Column '{col}' not found; subgroup may be empty.")
            return pd.Series(False, index=df0.index)
        if isinstance(rule, dict):
            if "contains_any" in rule:
                s = df0[col].astype(str).str.lower()
                anymask = np.zeros(len(df0), dtype=bool)
                for kw in rule["contains_any"]:
                    anymask |= s.str.contains(str(kw).lower(), na=False)
                mask &= anymask
            if "equals_any" in rule:
                s = df0[col].astype(str).str.strip().str.lower()
                vals = [str(v).strip().lower() for v in rule["equals_any"]]
                mask &= s.isin(vals)
            if "range" in rule:
                lo, hi = (rule["range"]+[None,None])[:2]
                x = pd.to_numeric(df0[col], errors="coerce")
                m = pd.Series(True, index=df0.index)
                if lo is not None: m &= (x >= lo)
                if hi is not None: m &= (x <= hi)
                mask &= m
        else:
            mask &= df0[col].astype(str).str.strip().str.lower().eq(str(rule).strip().lower())
    return mask

def run_subgroup(name, criteria):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "ALL"
    M = DF.copy()
    m = subgroup_mask(M, criteria)
    M = M[m].copy()
    n = len(M)
    if n == 0:
        print(f"[Skip] {name}: n=0 (no matches). See subgroup_catalog.csv for exact labels.")
        return
    if n < MIN_N_SUB and not ALLOW_SMALL:
        print(f"[Skip] {name}: n={n} < MIN_N_SUB={MIN_N_SUB}")
        return
    if n < MIN_N_SUB and ALLOW_SMALL:
        print(f"[Warn] {name}: running with small n={n} (<{MIN_N_SUB}). Interpret with caution.")

    rows=[]
    # Use robust ID name for outputs (keep your original ID column if present)
    id_cols_out = [c for c in ["patient_id","Patient ID"] if c in M.columns]
    for k, (_, r) in enumerate(M[["id_std","theta_S_vs_OBS","theta_SRT_vs_S","theta_SRT_vs_OBS"]].iterrows()):
        sim = simulate_three_arm(r["theta_S_vs_OBS"], r["theta_SRT_vs_S"], base_seed=200000+k)
        row = {
            "id_std": r["id_std"],
            "cost_OBS": sim["OBS"]["cost"],   "qaly_OBS": sim["OBS"]["qaly"],
            "cost_S":   sim["S"]["cost"],     "qaly_S":   sim["S"]["qaly"],
            "cost_SRT": sim["S+RT"]["cost"],  "qaly_SRT": sim["S+RT"]["qaly"],
            "death_m_OBS": sim["OBS"]["death_m"], "cause_OBS": sim["OBS"]["cause"],
            "death_m_S":   sim["S"]["death_m"],   "cause_S":   sim["S"]["cause"],
            "death_m_SRT": sim["S+RT"]["death_m"],"cause_SRT": sim["S+RT"]["cause"],
            "theta_S_vs_OBS": r["theta_S_vs_OBS"],
            "theta_SRT_vs_S": r["theta_SRT_vs_S"],
            "theta_SRT_vs_OBS": r["theta_SRT_vs_OBS"]
        }
        for c in id_cols_out:
            row[c] = M.loc[r.name, c]
        rows.append(row)
    res = pd.DataFrame(rows)

    # increments & ICERs
    res["dC_S_vs_OBS"]   = res["cost_S"]   - res["cost_OBS"]
    res["dQ_S_vs_OBS"]   = res["qaly_S"]   - res["qaly_OBS"]
    res["ICER_S_vs_OBS"] = np.where(res["dQ_S_vs_OBS"]>0, res["dC_S_vs_OBS"]/res["dQ_S_vs_OBS"], np.nan)
    res["dC_SRT_vs_S"]   = res["cost_SRT"] - res["cost_S"]
    res["dQ_SRT_vs_S"]   = res["qaly_SRT"] - res["qaly_S"]
    res["ICER_SRT_vs_S"] = np.where(res["dQ_SRT_vs_S"]>0, res["dC_SRT_vs_S"]/res["dQ_SRT_vs_S"], np.nan)
    res["dC_SRT_vs_OBS"] = res["cost_SRT"] - res["cost_OBS"]
    res["dQ_SRT_vs_OBS"] = res["qaly_SRT"] - res["qaly_OBS"]
    res["ICER_SRT_vs_OBS"] = np.where(res["dQ_SRT_vs_OBS"]>0, res["dC_SRT_vs_OBS"]/res["dQ_SRT_vs_OBS"], np.nan)

    pt_path  = os.path.join(BASE_DIR, f"msim3_patient_level_SUBGROUP={slug}.csv")
    res.to_csv(pt_path, index=False)
    summ = summarize_three(res, lambdas=WTP_LIST)
    sm_path = os.path.join(BASE_DIR, f"msim3_summary_SUBGROUP={slug}.csv")
    summ.to_csv(sm_path, index=False)

    pair, front, by = write_icer_tables(res, prefix=f"SUBGROUP={slug}")
    print(f"[Subgroup] {name}: n={n}")
    print(f"  Patient-level -> {pt_path}")
    print(f"  Summary       -> {sm_path}")
    print(f"  ICER pairwise -> {pair}")
    print(f"  ICER frontier -> {front}")
    print(f"  ICER by-treat -> {by}")

# ---------- Catalog of labels ----------
CATALOG_COLS = ["age_recode","grade_dcisset","tumor_size_recode","race_origin_recode","er","pr","registry"]
catalog = []
for c in CATALOG_COLS:
    if c in DF.columns:
        vc = (DF[c].fillna("Missing").value_counts(dropna=False)
              .reset_index().rename(columns={"index": c, c:"n"}))
        vc.insert(0, "column", c)
        catalog.append(vc)
if catalog:
    cat = pd.concat(catalog, ignore_index=True)
    cat_path = os.path.join(BASE_DIR, "subgroup_catalog.csv")
    cat.to_csv(cat_path, index=False)
    print(f"[Catalog] Wrote value counts -> {cat_path}")

# ---------- Always run 'AllEligible' ----------
run_subgroup("AllEligible", criteria={})

# ---------- Example subgroup (edit using subgroup_catalog.csv exact labels) ----------
example_criteria = {
    "race_origin_recode": {"contains_any": ["Non-Hispanic White","NHW"]},
    "age_recode": {"contains_any": ["65-69","65 to 69","65–69"]},
    "tumor_size_recode": {"contains_any": ["<=2","≤2","0-2","0–2","0-20","0 to 20","0–20","0-2 cm","0–2 cm"]},
}
run_subgroup("NHW_65to69_TumorSize_le2cm", example_criteria)

# ---------- Auto-run by level for key variables ----------
for col in AUTO_VARS:
    if col not in DF.columns:
        print(f"[Auto] Column {col} not found; skip.")
        continue
    vals = (DF[col].fillna("Missing").value_counts().index.tolist())
    for v in vals:
        run_subgroup(f"{col}=={v}", {col: {"equals_any":[v]}})

print("STEP-04 finished.")

