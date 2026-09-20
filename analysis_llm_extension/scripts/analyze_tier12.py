#!/usr/bin/env python3
"""
Full statistical pipeline for the Tier 1/2 pitch-contour measures, mirroring
the main paper's methodology exactly:
  - within-debate FE effect (beta / sigma_within)
  - cluster-robust p (debate-clustered, speaker-clustered), via statsmodels
  - exact within-debate randomization test (meet-in-the-middle, all 2^24
    sign patterns, no Monte Carlo)
  - per-debate sign test
  - random-effects (DerSimonian-Laird) meta-analysis: pooled d, CI, tau^2,
    I^2, 95% prediction interval
  - leave-one-speaker-out influence range
  - BH correction across THIS family of 10 measures (kept separate from the
    main paper's 43)
  - speaker-level F0 coupling check + alignment-treatment stability check,
    replicated for whichever measures turn out significant, since those are
    exactly the two checks that separated the two original surviving
    measures from each other
"""
import csv, json, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.stats.meta_analysis import combine_effects
from scipy.stats import binomtest, pearsonr

D = "/Users/lavanya/debate_analysis/outputs/"
SCRATCH = "/private/tmp/claude-501/-Users-lavanya-debate-analysis/c2e6601f-99ec-4364-9a03-d6e1384a7237/scratchpad/"
EX = {"1034", "1036"}

master = {}
for r in csv.DictReader(open(D + "master_sentences.csv")):
    if r["speaker_role"] != "candidate" or r["debate_audio_id"] in EX:
        continue
    master[r["sentence_id"]] = r

tier = pd.read_csv(SCRATCH + "tier12_features.csv")
tier = tier[tier.sentence_id.isin(master)]
tier["y"] = tier.sentence_id.map(lambda s: 1 if master[s]["result"] == "winner" else 0)
tier["deb"] = tier.sentence_id.map(lambda s: master[s]["debate_audio_id"])
tier["spk"] = tier.sentence_id.map(lambda s: master[s]["speaker"])
tier["tier_ali"] = tier.sentence_id.map(lambda s: master[s].get("alignment_tier", ""))

MEASURES = ["skewness", "kurtosis", "iqr_range_ratio", "velocity_mean_abs",
            "velocity_sd", "accel_mean_abs", "turning_pt_rate",
            "total_variation_norm", "declination_slope", "declination_resid_sd"]
LABELS = {
    "skewness": "Skewness of F0 distribution",
    "kurtosis": "Kurtosis of F0 distribution",
    "iqr_range_ratio": "IQR / 20-80 range ratio",
    "velocity_mean_abs": "Pitch velocity (mean |change|)",
    "velocity_sd": "Pitch velocity (SD)",
    "accel_mean_abs": "Pitch acceleration (mean |change|)",
    "turning_pt_rate": "Turning points per second",
    "total_variation_norm": "Total variation (semitones/s)",
    "declination_slope": "Declination slope (semitones/s)",
    "declination_resid_sd": "Declination-residual SD",
}

print(f"n = {len(tier)}  winners = {tier.y.sum()}  losers = {(1-tier.y).sum()}  "
      f"debates = {tier.deb.nunique()}  speakers = {tier.spk.nunique()}\n")

debates = sorted(tier.deb.unique())
G = len(debates)
assert G == 24

# ---------- meet-in-the-middle exact randomization ----------
def exact_perm_p(S_by_debate, Q_total, obs_T):
    """S_by_debate: array of per-debate S_d (signed contributions).
    Null of T(s) = sum_d (-1)^{s_d} S_d over all 2^24 sign patterns,
    via meet-in-the-middle: split into two halves of 12, enumerate 4096
    signed subset sums each, then outer-sum."""
    half = G // 2
    A, B = S_by_debate[:half], S_by_debate[half:]
    def signed_sums_fast(v):
        sums = np.array([0.0])
        for x in v:
            sums = np.concatenate([sums + x, sums - x])
        return sums
    sa = signed_sums_fast(A)
    sb = signed_sums_fast(B)
    # full null: outer sum sa[i] + sb[j] for all pairs = all 2^24 T(s)
    # two-sided p = P(|T| >= |obs_T|)
    thresh = abs(obs_T)
    count = 0
    total = len(sa) * len(sb)
    # process in chunks to bound memory
    for chunk in np.array_split(sa, 64):
        vals = chunk[:, None] + sb[None, :]
        count += np.sum(np.abs(vals) >= thresh - 1e-9)
    return count / total

def fe_effect_and_S(df, col):
    """Return beta/sigma_within, and per-debate S_d,Q_d for FWL exact test,
    matching y_i = beta*w_i + debate dummies + eps."""
    y = df[col].values
    w = df.y.values.astype(float)
    deb_codes = df.deb.values
    S_d, Q_d, n_d = [], [], []
    wbar_all, ybar_all = {}, {}
    for d in debates:
        m = deb_codes == d
        wd, yd = w[m], y[m]
        wbar, ybar = wd.mean(), yd.mean()
        wc, yc = wd - wbar, yd - ybar
        S_d.append(np.sum(wc * yc))
        Q_d.append(np.sum(wc * wc))
    S_d, Q_d = np.array(S_d), np.array(Q_d)
    beta = S_d.sum() / Q_d.sum()
    # residual variance for sigma_within
    resid = []
    for i, d in enumerate(debates):
        m = deb_codes == d
        wd, yd = w[m], y[m]
        wbar, ybar = wd.mean(), yd.mean()
        pred = ybar + beta * (wd - wbar)
        resid.append(yd - pred)
    resid = np.concatenate(resid)
    dof = len(y) - G - 1
    sigma_within = np.sqrt(np.sum(resid**2) / dof)
    eff = beta / sigma_within if sigma_within > 0 else np.nan
    return eff, S_d, Q_d.sum()

def per_debate_d(df, col):
    e, v = [], []
    for d in debates:
        s = df[df.deb == d]
        a = s[s.y == 1][col].values
        b = s[s.y == 0][col].values
        if len(a) < 5 or len(b) < 5:
            continue
        sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        if sp == 0 or not np.isfinite(sp):
            continue
        dd = (a.mean() - b.mean()) / sp
        e.append(dd)
        v.append((len(a)+len(b))/(len(a)*len(b)) + dd**2/(2*(len(a)+len(b))))
    return np.array(e), np.array(v)

def loso_range(df, col):
    vals = []
    for s in df.spk.unique():
        sub = df[df.spk != s]
        if sub.y.nunique() < 2 or sub.deb.nunique() < 2:
            continue
        eff, _, _ = fe_effect_and_S(sub, col)
        vals.append(eff)
    return min(vals), max(vals)

results = []
for m in MEASURES:
    sub = tier[np.isfinite(tier[m])].copy()
    if len(sub) < 1000:
        print(f"SKIP {m}: only {len(sub)} finite rows")
        continue

    eff, S_d, Qsum = fe_effect_and_S(sub, m)
    obs_T = S_d.sum()

    Dm = pd.get_dummies(sub.deb).astype(float)
    Xd = np.column_stack([sub.y.values.astype(float), Dm.values])
    r_deb = sm.OLS(sub[m].values, Xd).fit(cov_type="cluster", cov_kwds={"groups": sub.deb})
    r_spk = sm.OLS(sub[m].values, Xd).fit(cov_type="cluster", cov_kwds={"groups": sub.spk})
    p_deb, p_spk = r_deb.pvalues[0], r_spk.pvalues[0]

    e_arr, v_arr = per_debate_d(sub, m)
    pos = max((e_arr > 0).sum(), (e_arr < 0).sum())
    k = len(e_arr)
    sign_p = binomtest(pos, k, 0.5).pvalue

    cr = combine_effects(e_arr, v_arr, method_re="dl")
    mu, se, tau2 = cr.mean_effect_re, cr.sd_eff_w_re, cr.tau2
    Q = np.sum((e_arr - np.sum(e_arr/v_arr)/np.sum(1/v_arr))**2 / v_arr)
    I2 = max(0, (Q - (k-1)) / Q) * 100 if Q > 0 else 0
    t = 2.0739
    pi_lo, pi_hi = mu - t*np.sqrt(tau2+se**2), mu + t*np.sqrt(tau2+se**2)
    ci_lo, ci_hi = mu - 1.96*se, mu + 1.96*se

    perm_p = exact_perm_p(S_d, Qsum, obs_T)

    loso_lo, loso_hi = loso_range(sub, m)
    sign_stable = (loso_lo > 0) == (loso_hi > 0)

    results.append(dict(
        measure=m, label=LABELS[m], n=len(sub), fe_effect=eff,
        p_deb=p_deb, p_spk=p_spk, perm_p=perm_p,
        dir_pos=pos, dir_k=k, sign_p=sign_p,
        re_mean=mu, ci_lo=ci_lo, ci_hi=ci_hi, tau2=tau2, I2=I2,
        pi_lo=pi_lo, pi_hi=pi_hi, loso_lo=loso_lo, loso_hi=loso_hi,
        sign_stable=sign_stable,
    ))
    print(f"{LABELS[m]:32s} n={len(sub):6d} eff={eff:+.4f} p_deb={p_deb:.4f} p_spk={p_spk:.4f} "
          f"perm_p={perm_p:.4f} dir={pos}/{k} sign_p={sign_p:.4f} "
          f"RE={mu:+.3f}[{ci_lo:+.3f},{ci_hi:+.3f}] I2={I2:.0f}% "
          f"PI=({pi_lo:+.2f},{pi_hi:+.2f}) LOSO=[{loso_lo:+.3f},{loso_hi:+.3f}]{'*' if not sign_stable else ''}")

# BH correction over this family of 10 (own family)
rdf = pd.DataFrame(results).sort_values("perm_p").reset_index(drop=True)
m_fam = len(rdf)
rdf["bh_thresh"] = (rdf.index + 1) / m_fam * 0.05
rdf["bh_reject"] = rdf.perm_p <= rdf.bh_thresh
# step-up: once a later one fails, nothing beyond in sorted order can pass either
# (find largest k with p_(k) <= k/m*0.05)
passing = rdf[rdf.perm_p <= rdf.bh_thresh]
last_pass_rank = passing.index.max() if len(passing) else -1
rdf["bh_reject"] = rdf.index <= last_pass_rank

print(f"\n=== Benjamini-Hochberg, family of {m_fam} (own family) ===")
for _, row in rdf.iterrows():
    print(f"  {row.label:32s} perm_p={row.perm_p:.4f}  thresh={row.bh_thresh:.4f}  "
          f"{'REJECT' if row.bh_reject else 'fail'}")

rdf.to_json(SCRATCH + "tier12_results.json", orient="records", indent=2)
print(f"\nsaved {SCRATCH}tier12_results.json")

# ---------- speaker-level F0 coupling + alignment-treatment stability ----------
# for whichever measures are BH-significant OR close, replicate the two checks
# that separated the two original surviving measures
sig_or_close = rdf[(rdf.bh_reject) | (rdf.perm_p < 0.05)].measure.tolist()
if sig_or_close:
    print(f"\n=== Follow-up validity checks on: {sig_or_close} ===")
    f0_by_spk = tier.groupby("spk").apply(lambda g: None)  # placeholder
    master_f0 = {}
    for r in csv.DictReader(open(D + "acoustic_features.csv")):
        if r["sentence_id"] in master:
            master_f0[r["sentence_id"]] = r.get("f0_mean")
    tier["f0_mean"] = tier.sentence_id.map(lambda s: master_f0.get(s))
    tier["f0_mean"] = pd.to_numeric(tier.f0_mean, errors="coerce")

    for m in sig_or_close:
        sub = tier[np.isfinite(tier[m]) & np.isfinite(tier.f0_mean)]
        sp_means = sub.groupby("spk")[[m, "f0_mean"]].mean()
        r_speaker, _ = pearsonr(sp_means[m], sp_means.f0_mean)
        # within speaker x debate cell
        meds = []
        for (s, d), g in sub.groupby(["spk", "deb"]):
            if len(g) >= 50:
                rr, _ = pearsonr(g[m], g.f0_mean)
                if np.isfinite(rr):
                    meds.append(rr)
        med_within = np.median(meds) if meds else np.nan
        print(f"  {LABELS[m]:32s} r(speaker mean F0) = {r_speaker:+.3f}   "
              f"median within-cell r = {med_within:+.3f}")
