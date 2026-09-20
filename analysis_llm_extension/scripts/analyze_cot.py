#!/usr/bin/env python3
"""
CoT vs. Direct prompting, agreement with the rule-based tagger, on the
identical stratified sample (paper extension item #2). Reuses the exact
agreement machinery (PSA/kappa/PABAK/AC1, MASI-distance Krippendorff alpha)
already validated for the main-paper rule-vs-direct-LLM comparison
(p5_agree.py / common.py).
"""
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(10 ** 7)

PROJECT = Path("/Users/lavanya/debate_analysis")
HERE = Path(__file__).resolve().parent

DAMSL_MAP = [("damsl_Q_W", "Q-W"), ("damsl_YNQ", "YNQ"), ("damsl_Q_RHET", "Q-RHET"),
             ("damsl_OQ", "OQ"), ("damsl_SEEP", "SEEP"), ("damsl_S", "S"),
             ("damsl_S_Inform", "S-Inform"), ("damsl_IAFA", "IAFA"), ("damsl_IAFA_OF", "IAFA-OF"),
             ("damsl_ACK", "ACK"), ("damsl_AGR", "AGR"), ("damsl_DIS", "DIS"), ("damsl_REJ", "REJ"),
             ("damsl_ANS", "ANS"), ("damsl_GR", "GR"), ("damsl_APO", "APO"), ("damsl_THK", "THK"),
             ("damsl_EXPL", "EXPL"), ("damsl_CORR", "CORR"), ("damsl_CH", "CH"), ("damsl_TT", "TT"),
             ("damsl_TG", "TG"), ("damsl_T_REQ", "T-REQ"), ("damsl_R_REQ", "R-REQ"),
             ("damsl_MS", "MS"), ("damsl_HS", "HS"), ("damsl_INT", "INT"), ("damsl_TA", "TA")]
BEADS_MAP = [("bias_AF", "AF"), ("bias_AE", "AE"), ("bias_AP", "AP"), ("bias_APAT", "APAT"),
             ("bias_UF", "UF"), ("bias_DF", "DF"), ("bias_PB", "PB"), ("bias_PER", "PER"),
             ("bias_IT", "IT"), ("bias_IP", "IP"), ("bias_BQ", "BQ"), ("bias_ATTR", "ATTR"),
             ("bias_AEX", "AEX"), ("bias_REB", "REB"), ("bias_RB", "RB"), ("bias_GB", "GB"),
             ("bias_GD", "GD"), ("bias_CB", "CB"), ("bias_IA", "IA"), ("bias_SE", "SE"),
             ("bias_CBias", "CBias")]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agreement(rule, other):
    a = b = c = d = 0
    for r, o in zip(rule, other):
        if r and o: a += 1
        elif r and not o: b += 1
        elif o and not r: c += 1
        else: d += 1
    n = a + b + c + d
    po = (a + d) / n if n else float("nan")
    p1r, p1o = (a + b) / n, (a + c) / n
    pe = p1r * p1o + (1 - p1r) * (1 - p1o)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    psa = (2 * a) / (2 * a + b + c) if (2 * a + b + c) else float("nan")
    pabak = 2 * po - 1
    pi = (p1r + p1o) / 2.0
    pe_g = 2 * pi * (1 - pi)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else float("nan")
    return {"n": n, "po": po, "kappa": kappa, "psa": psa, "pabak": pabak, "ac1": ac1,
            "n_rule": a + b, "n_other": a + c}


def masi(a, b):
    if a == b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    j = len(a & b) / len(union)
    inter = a & b
    if not inter:
        m = 0.0
    elif a <= b or b <= a:
        m = 2 / 3
    else:
        m = 1 / 3
    return 1.0 - j * m


def kripp_alpha(items):
    num_o = sum(masi(va, vb) for va, vb in items)
    den_o = len(items)
    do = num_o / den_o
    bag = Counter()
    for va, vb in items:
        bag[va] += 1; bag[vb] += 1
    vals = list(bag.items())
    N = sum(bag.values())
    num_e = 0.0
    for i in range(len(vals)):
        vi, ni = vals[i]
        for j2 in range(i + 1, len(vals)):
            vj, nj = vals[j2]
            num_e += ni * nj * masi(vi, vj)
    de = (2 * num_e) / (N * (N - 1))
    return (1.0 - do / de if de else float("nan")), do, de


def tagset(row, field):
    return frozenset(t.strip().upper() for t in (row.get(field) or "").split(";") if t.strip())


def main():
    sample_ids = [r["sentence_id"] for r in load_csv(HERE / "cot_sample_ids.csv")]
    sample_set = set(sample_ids)
    rule = {r["sentence_id"]: r for r in load_csv(PROJECT / "outputs" / "damsl_bias_tags.csv")
            if r["sentence_id"] in sample_set}
    direct = {r["sentence_id"]: r for r in load_csv(PROJECT / "outputs" / "llm_tags.csv")
              if r["sentence_id"] in sample_set}
    cot = {r["sentence_id"]: r for r in load_csv(HERE / "cot_tags.csv")
           if r["sentence_id"] in sample_set}

    ids = [s for s in sample_ids if s in rule and s in direct and s in cot]
    print(f"[n] {len(ids)} sentences with rule + direct + CoT tags")

    def sets_for(store, sid, field, use_llm_field):
        if store is rule:
            return frozenset(c for col, c in (DAMSL_MAP if field == "damsl" else BEADS_MAP)
                              if store[sid].get(col) == "1")
        else:
            return tagset(store[sid], use_llm_field)

    results = {"n": len(ids)}

    for phase, field, mapping in (("damsl", "damsl_tags", DAMSL_MAP), ("beads", "beads_tags", BEADS_MAP)):
        rule_sets = {s: sets_for(rule, s, "damsl" if field == "damsl_tags" else "beads", None) for s in ids}
        direct_sets = {s: tagset(direct[s], field) for s in ids}
        cot_sets = {s: tagset(cot[s], field) for s in ids}

        for cmp_name, other_sets in (("direct_vs_rule", direct_sets), ("cot_vs_rule", cot_sets)):
            items = [(rule_sets[s], other_sets[s]) for s in ids]
            alpha, do, de = kripp_alpha(items)
            results.setdefault(phase, {})[cmp_name] = {"alpha_masi": alpha, "D_o": do, "D_e": de}

        items_cd = [(cot_sets[s], direct_sets[s]) for s in ids]
        alpha_cd, do_cd, de_cd = kripp_alpha(items_cd)
        results.setdefault(phase, {})["cot_vs_direct"] = {"alpha_masi": alpha_cd, "D_o": do_cd, "D_e": de_cd}

        # per-category agreement (top categories by rule count, matching tab:kappa style)
        per_cat = {}
        for col, code in mapping:
            r01 = [1 if col in [] else (1 if code in rule_sets[s] else 0) for s in ids]
            d01 = [1 if code in direct_sets[s] else 0 for s in ids]
            c01 = [1 if code in cot_sets[s] else 0 for s in ids]
            n_rule, n_direct, n_cot = sum(r01), sum(d01), sum(c01)
            if n_rule == 0 and n_direct == 0 and n_cot == 0:
                continue
            per_cat[code] = {
                "n_rule": n_rule, "n_direct": n_direct, "n_cot": n_cot,
                "direct_vs_rule": agreement(r01, d01),
                "cot_vs_rule": agreement(r01, c01),
            }
        results[phase]["per_category"] = per_cat

    # tagging-rate comparison
    def rate(store, field):
        return sum(1 for s in ids if tagset(store[s], field)) / len(ids)

    results["tagging_rates"] = {
        "direct_damsl": rate(direct, "damsl_tags"), "cot_damsl": rate(cot, "damsl_tags"),
        "direct_beads": rate(direct, "beads_tags"), "cot_beads": rate(cot, "beads_tags"),
    }

    def avg_n(store, field):
        return sum(int(store[s].get("n_" + field.split("_")[0] + "_tags", 0) or 0) for s in ids) / len(ids)

    out = HERE / "cot_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[write] {out}")

    print("\n=== Whole-set Krippendorff alpha (MASI) ===")
    for phase in ("damsl", "beads"):
        print(f"  {phase}:")
        for k in ("direct_vs_rule", "cot_vs_rule", "cot_vs_direct"):
            v = results[phase][k]
            print(f"    {k:16s} alpha={v['alpha_masi']:+.4f}  D_o={v['D_o']:.4f} D_e={v['D_e']:.4f}")

    print("\n=== Tagging rates (fraction of sample sentences with >=1 tag) ===")
    print(results["tagging_rates"])


if __name__ == "__main__":
    main()
