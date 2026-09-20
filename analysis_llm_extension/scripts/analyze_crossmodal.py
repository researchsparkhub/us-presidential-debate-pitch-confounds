#!/usr/bin/env python3
"""
Analysis for Method 4 (LLM cross-modal reasoning), the explainability/
grounding check, and the contamination probe.

Outputs crossmodal_results.json, consumed by the paper-writing pass.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
PRED = HERE / "crossmodal_predictions.jsonl"
PROBE = HERE / "crossmodal_contamination.jsonl"
CACHE = HERE / "llm_ext_cache"

FRIENDLY_NAMES = json.loads((HERE / "names.json").read_text()) if (HERE / "names.json").exists() else None


def load_jsonl(p):
    return [json.loads(l) for l in p.open()]


def null_pmf(n_candidates_per_debate):
    """Exact distribution of #correct under independent random guessing,
    one debate at a time, chance = 1/n_candidates for that debate."""
    pmf = np.array([1.0])
    for n in n_candidates_per_debate:
        p = 1.0 / n
        pmf = np.convolve(pmf, [1 - p, p])
    return pmf  # pmf[k] = P(exactly k correct)


def upper_tail_p(pmf, observed):
    return float(pmf[observed:].sum())


def main():
    preds = load_jsonl(PRED)
    probes = load_jsonl(PROBE)
    debates = {p.stem: json.loads(p.read_text()) for p in CACHE.glob("*.json") if p.stem != "_manifest"}
    n_cand = {d: len(debates[d]["candidates"]) for d in debates}
    debate_ids = sorted(debates, key=int)

    results = {"n_debates": len(debate_ids), "n_candidates_per_debate": n_cand}

    # ---------------- 1. Per-condition accuracy ----------------
    by_cond = defaultdict(list)
    for r in preds:
        by_cond[r["condition"]].append(r)

    cond_results = {}
    baseline_pmf = null_pmf([n_cand[d] for d in debate_ids])
    for cond in ("transcript", "descriptors", "both"):
        rows = by_cond[cond]
        # trial-level (every repeat counted separately)
        trial_correct = sum(1 for r in rows if r["predicted_letter"] == r["winner_letter"])
        trial_n = len(rows)
        # debate-level majority vote across repeats
        by_deb = defaultdict(list)
        for r in rows:
            by_deb[r["debate_id"]].append(r)
        maj_correct = 0
        unanimous = 0
        calls = {}
        for d, rs in by_deb.items():
            preds_for_d = [r["predicted_candidate"] for r in rs]
            vote = Counter(preds_for_d).most_common(1)[0][0]
            calls[d] = vote
            if vote == debates[d]["winner"]:
                maj_correct += 1
            if len(set(preds_for_d)) == 1:
                unanimous += 1
        n_deb = len(by_deb)
        pt = binomtest(trial_correct, trial_n, 0.5, alternative="greater")
        p_majority = upper_tail_p(baseline_pmf, maj_correct)
        expected_chance = float(np.dot([1.0 / n_cand[d] for d in debate_ids], np.ones(len(debate_ids))) / len(debate_ids))
        cond_results[cond] = {
            "trial_correct": trial_correct, "trial_n": trial_n,
            "trial_accuracy": trial_correct / trial_n,
            "trial_p_vs_half": pt.pvalue,
            "debate_majority_correct": maj_correct, "debate_n": n_deb,
            "debate_majority_accuracy": maj_correct / n_deb,
            "debate_majority_p_vs_chance": p_majority,
            "expected_chance_accuracy": expected_chance,
            "unanimous_debates": unanimous,
            "unanimity_rate": unanimous / n_deb,
            "calls": calls,
        }
    results["conditions"] = cond_results

    # BH correction across the 3 headline condition-vs-chance tests
    pvals = [cond_results[c]["debate_majority_p_vs_chance"] for c in ("transcript", "descriptors", "both")]
    order = np.argsort(pvals)
    m = len(pvals)
    bh = [None] * m
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = m - rank
        val = pvals[idx] * m / i
        prev = min(prev, val)
        bh[idx] = min(prev, 1.0)
    for c, p in zip(("transcript", "descriptors", "both"), bh):
        cond_results[c]["debate_majority_p_bh"] = p

    # ---------------- 2. Pairwise agreement between conditions (paired, debate level) ----------------
    pairwise = {}
    conds = ("transcript", "descriptors", "both")
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            c1, c2 = conds[i], conds[j]
            calls1, calls2 = cond_results[c1]["calls"], cond_results[c2]["calls"]
            agree = sum(1 for d in debate_ids if calls1.get(d) == calls2.get(d))
            pairwise[f"{c1}_vs_{c2}"] = {"agree": agree, "n": len(debate_ids), "rate": agree / len(debate_ids)}
    results["pairwise_agreement"] = pairwise

    # ---------------- 3. Position bias: does the model over-pick "A"? ----------------
    letter_counts = defaultdict(Counter)
    for r in preds:
        letter_counts[r["condition"]][r["predicted_letter"]] += 1
    results["position_bias"] = {c: dict(letter_counts[c]) for c in letter_counts}

    # ---------------- 4. Contamination probe ----------------
    def name_match(guess_list, real_list):
        g = " ".join(guess_list).lower()
        hits = 0
        for real in real_list:
            surname = real.split()[-1].lower()
            if surname in g:
                hits += 1
        return hits

    probe_results = {}
    for cond in ("transcript", "descriptors"):
        rows = [p for p in probes if p["condition"] == cond]
        recognized = sum(1 for p in rows if p["recognized"])
        year_correct = sum(1 for p in rows if p.get("guessed_year") and str(p["year"]) in str(p["guessed_year"]))
        name_hits = [name_match(p.get("guessed_candidates") or [], p["real_candidates"]) for p in rows]
        at_least_one_name = sum(1 for h in name_hits if h >= 1)
        both_names = sum(1 for h in name_hits if h >= 2)
        probe_results[cond] = {
            "n": len(rows), "self_reported_recognized": recognized,
            "year_correct": year_correct,
            "at_least_one_candidate_named": at_least_one_name,
            "both_candidates_named": both_names,
        }
    results["contamination_probe"] = probe_results

    # ---------------- 5. Grounding check: cited_evidence vs real z-scored data ----------------
    def real_direction(debate_id, measure, letter_map):
        d = debates[debate_id]
        vals = {letter_map[c]: d["descriptors"][c].get(measure) for c in d["candidates"]}
        if any(v is None for v in vals.values()):
            return None
        items = sorted(vals.items(), key=lambda kv: -kv[1])
        top_val = items[0][1]
        # "similar" if top two are within 0.15 SD of each other
        if len(items) > 1 and abs(items[0][1] - items[1][1]) < 0.15:
            return "similar"
        return items[0][0]

    grounding = {"descriptors": {"checked": 0, "consistent": 0}, "both": {"checked": 0, "consistent": 0}}
    for r in preds:
        if r["condition"] not in ("descriptors", "both") or r["mode"] != "measures":
            continue
        for cite in (r["cited_evidence"] or []):
            item = cite["item"]
            if item == "other":
                continue
            real = real_direction(r["debate_id"], item, r["letter_map"])
            if real is None:
                continue
            grounding[r["condition"]]["checked"] += 1
            if cite["favors"] == real:
                grounding[r["condition"]]["consistent"] += 1
    for c in grounding:
        chk = grounding[c]["checked"]
        grounding[c]["consistency_rate"] = grounding[c]["consistent"] / chk if chk else None
    results["grounding_check"] = grounding

    # ---------------- 6. Fabrication scan: transcript-only condition, no numbers given ----------------
    NUM_PATTERN = re.compile(
        r"\b\d+(\.\d+)?\s*(hz|hertz|db|decibel|semitone|jitter|shimmer|wpm|"
        r"words per minute|ms|milliseconds?|percentile|z-score|standard deviation)s?\b",
        re.IGNORECASE,
    )
    fab_hits = []
    for r in preds:
        if r["condition"] != "transcript":
            continue
        if NUM_PATTERN.search(r["explanation"] or ""):
            fab_hits.append({"debate_id": r["debate_id"], "repeat": r["repeat"],
                              "explanation": r["explanation"]})
    results["fabrication_scan"] = {"n_transcript_explanations": len(by_cond["transcript"]),
                                    "n_flagged": len(fab_hits), "examples": fab_hits[:5]}

    # ---------------- 7. Cited-evidence theme frequency (transcript condition) ----------------
    theme_counts = Counter()
    for r in preds:
        if r["condition"] == "transcript":
            for cite in (r["cited_evidence"] or []):
                theme_counts[cite["item"]] += 1
    results["transcript_theme_frequency"] = dict(theme_counts.most_common())

    # ---------------- 8. Cited-measure frequency (descriptors/both) ----------------
    measure_counts = Counter()
    for r in preds:
        if r["condition"] in ("descriptors", "both"):
            for cite in (r["cited_evidence"] or []):
                measure_counts[cite["item"]] += 1
    results["descriptor_citation_frequency"] = dict(measure_counts.most_common())

    out = HERE / "crossmodal_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[write] {out}")

    print("\n=== Summary ===")
    for c in ("transcript", "descriptors", "both"):
        cr = cond_results[c]
        print(f"{c:12s}  trial acc {cr['trial_accuracy']:.3f} (p={cr['trial_p_vs_half']:.4f})  "
              f"debate-majority {cr['debate_majority_correct']}/{cr['debate_n']} "
              f"(p={cr['debate_majority_p_vs_chance']:.4f}, BH={cr['debate_majority_p_bh']:.4f})  "
              f"unanimity {cr['unanimity_rate']:.2f}")
    print("\nPairwise agreement:", pairwise)
    print("\nContamination probe:", probe_results)
    print("\nGrounding check:", grounding)
    print("\nFabrication scan:", results["fabrication_scan"]["n_flagged"], "/",
          results["fabrication_scan"]["n_transcript_explanations"])


if __name__ == "__main__":
    main()
