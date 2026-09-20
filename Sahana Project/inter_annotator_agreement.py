#!/usr/bin/env python3
"""
Inter-annotator agreement for the Sahana Project verification task.

Reads the verified per-debate CSVs (exported back from Google Sheets) and reports
how much the annotators agree, separately for Phase A (DAMSL) and Phase B (Beads).
Tags are multi-label (semicolon-separated sets), so it uses set-aware metrics.

METRICS (per phase, pooled across all debates):
  - Exact-match agreement  : % of sentences where an annotator pair's tag SETS match exactly
  - Mean Jaccard           : average set overlap |A∩B| / |A∪B| across pairs
  - Krippendorff's alpha    : chance-corrected reliability using MASI set distance
                              (>0.8 good, 0.67-0.8 tentative, <0.67 weak)
  - Per-tag Fleiss' kappa   : reliability of each individual code (flags contentious ones)

Also writes a disagreement-hotspots CSV (sentences the annotators most diverge on)
for adjudication.

No third-party libraries required.

Usage:
    # 1. In Google Sheets, File -> Download -> CSV for each verified debate,
    #    and drop the files into  Sahana Project/verified/
    python "Sahana Project/inter_annotator_agreement.py"

    # or point at any folder of verified CSVs:
    python "Sahana Project/inter_annotator_agreement.py" --input path/to/csvs

    # include sentences where all annotators left a phase blank (agreement-on-empty):
    python "Sahana Project/inter_annotator_agreement.py" --include-empty
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "verified"
OUTDIR = HERE / "agreement"
PHASES = {"A": "Phase A (DAMSL)", "B": "Phase B (Beads)"}


# ---------- parsing ----------------------------------------------------------
def parse_cell(cell: str) -> frozenset:
    """A tag cell -> frozenset of normalized codes ('S; ATTR' -> {'S','ATTR'})."""
    return frozenset(t.strip().upper() for t in (cell or "").split(";") if t.strip())


def annotator_columns(header):
    """Map annotator number -> {'A': col_idx, 'B': col_idx} from the header."""
    cols = defaultdict(dict)
    for idx, col in enumerate(header):
        m = re.search(r"Annotator\s+(\d+)", col, re.IGNORECASE)
        if not m:
            continue
        a = int(m.group(1))
        if "DAMSL" in col.upper():
            cols[a]["A"] = idx
        elif "BEADS" in col.upper():
            cols[a]["B"] = idx
    return dict(sorted(cols.items()))


def load(input_dir, include_empty):
    """Return {phase: list of {'file','unit','text','ratings':{ann:frozenset}}}."""
    files = sorted(glob.glob(str(Path(input_dir) / "*.csv")))
    if not files:
        return None, []
    data = {"A": [], "B": []}
    annotators = None
    for fp in files:
        with open(fp, newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header:
                continue
            cols = annotator_columns(header)
            if len(cols) < 2:
                continue
            annotators = sorted(cols)
            fname = Path(fp).stem
            for row in r:
                if not row or not any(row):
                    continue
                unit = row[0] if len(row) > 0 else ""
                text = row[2] if len(row) > 2 else ""
                for ph in ("A", "B"):
                    ratings = {}
                    for a in annotators:
                        ci = cols[a].get(ph)
                        if ci is not None and ci < len(row):
                            ratings[a] = parse_cell(row[ci])
                    nonempty = sum(1 for s in ratings.values() if s)
                    # include a row only if it looks reviewed for this phase
                    if not include_empty and nonempty < 2:
                        continue
                    if len(ratings) >= 2:
                        data[ph].append({"file": fname, "unit": unit, "text": text, "ratings": ratings})
    return annotators, data


# ---------- set distance (MASI) ----------------------------------------------
def masi_distance(a: frozenset, b: frozenset) -> float:
    """1 - MASI similarity. 0 = identical sets, 1 = disjoint."""
    if a == b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    j = len(a & b) / len(union)
    if not (a & b):
        m = 0.0                       # disjoint
    elif a <= b or b <= a:
        m = 2 / 3                     # one is a subset of the other
    else:
        m = 1 / 3                     # overlap, neither a subset
    return 1.0 - j * m


# ---------- metrics ----------------------------------------------------------
def pairwise_stats(items, annotators):
    """Exact-match % and mean Jaccard, averaged over annotator pairs."""
    exact, jacc, n = [], [], 0
    for a, b in combinations(annotators, 2):
        em = jc = cnt = 0
        for it in items:
            ra, rb = it["ratings"].get(a), it["ratings"].get(b)
            if ra is None or rb is None:
                continue
            cnt += 1
            if ra == rb:
                em += 1
            union = ra | rb
            jc += 1.0 if not union else len(ra & rb) / len(union)
        if cnt:
            exact.append(em / cnt)
            jacc.append(jc / cnt)
            n = max(n, cnt)
    if not exact:
        return None
    return {"exact": sum(exact) / len(exact), "jaccard": sum(jacc) / len(jacc), "n": len(items)}


def krippendorff_alpha(items):
    """Krippendorff's alpha with MASI distance (pairwise formulation)."""
    # observed disagreement
    num_o = den_o = 0.0
    bag = Counter()
    for it in items:
        vals = list(it["ratings"].values())
        m = len(vals)
        if m < 2:
            continue
        pair_sum = sum(masi_distance(vals[i], vals[j])
                       for i in range(m) for j in range(i + 1, m))
        num_o += pair_sum / (m - 1)
        den_o += m / 2.0
        for v in vals:
            bag[v] += 1
    if den_o == 0:
        return None
    do = num_o / den_o
    # expected disagreement from the global bag of ratings
    vals = list(bag.items())
    N = sum(bag.values())
    if N < 2:
        return None
    num_e = 0.0
    for i in range(len(vals)):
        vi, ni = vals[i]
        for j in range(i + 1, len(vals)):
            vj, nj = vals[j]
            num_e += ni * nj * masi_distance(vi, vj)
    de = (2 * num_e) / (N * (N - 1)) if N > 1 else 0.0
    if de == 0:
        return 1.0 if do == 0 else 0.0
    return 1.0 - do / de


def fleiss_per_tag(items, annotators):
    """Fleiss' kappa per individual code (present/absent), plus support."""
    n = len(annotators)
    codes = set()
    for it in items:
        for s in it["ratings"].values():
            codes |= s
    out = {}
    for code in sorted(codes):
        rows = []            # count of annotators marking `code` present, per sentence
        for it in items:
            r = it["ratings"]
            if len(r) < n:   # need all annotators for a clean Fleiss row
                continue
            rows.append(sum(1 for s in r.values() if code in s))
        if len(rows) < 2:
            continue
        support = sum(1 for k in rows if k > 0)
        # Fleiss kappa, 2 categories (present/absent)
        p_i = [((k * k + (n - k) * (n - k)) - n) / (n * (n - 1)) for k in rows]
        p_bar = sum(p_i) / len(p_i)
        p_present = sum(rows) / (len(rows) * n)
        p_e = p_present ** 2 + (1 - p_present) ** 2
        kappa = None if p_e >= 1.0 else (p_bar - p_e) / (1 - p_e)
        out[code] = {"kappa": kappa, "support": support, "n": len(rows)}
    return out


def disagreements(items):
    """Rows sorted by mean pairwise MASI distance, descending (worst first)."""
    scored = []
    for it in items:
        vals = list(it["ratings"].values())
        m = len(vals)
        if m < 2:
            continue
        pairs = [masi_distance(vals[i], vals[j]) for i in range(m) for j in range(i + 1, m)]
        d = sum(pairs) / len(pairs)
        if d > 0:
            scored.append((d, it))
    scored.sort(key=lambda x: -x[0])
    return scored


# ---------- report -----------------------------------------------------------
def label(alpha):
    if alpha is None:
        return ""
    if alpha >= 0.8:
        return "good"
    if alpha >= 0.667:
        return "tentative"
    return "weak"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="folder of verified CSVs")
    ap.add_argument("--include-empty", action="store_true",
                    help="also count rows where <2 annotators tagged a phase (agreement-on-empty)")
    ap.add_argument("--top", type=int, default=400, help="max disagreement rows to export per phase")
    args = ap.parse_args()

    annotators, data = load(args.input, args.include_empty)
    if annotators is None:
        print(f"No verified CSVs found in {args.input}\n"
              f"  -> In Google Sheets: File > Download > CSV for each debate, put them in that folder.")
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Annotators detected: {annotators}   "
          f"(rows: Phase A = {len(data['A']):,}, Phase B = {len(data['B']):,}; "
          f"blank-included = {args.include_empty})\n")

    summary_rows = []
    for ph in ("A", "B"):
        items = data[ph]
        print(f"=== {PHASES[ph]} ===")
        if not items:
            print("  (no reviewed rows yet)\n")
            continue
        ps = pairwise_stats(items, annotators)
        alpha = krippendorff_alpha(items)
        print(f"  sentences compared : {ps['n']:,}")
        print(f"  exact-set match    : {ps['exact']*100:5.1f}%  (avg over annotator pairs)")
        print(f"  mean Jaccard       : {ps['jaccard']:.3f}")
        print(f"  Krippendorff alpha : {alpha:.3f}  ({label(alpha)})" if alpha is not None else "  Krippendorff alpha : n/a")
        summary_rows.append({"phase": PHASES[ph], "n": ps["n"],
                             "exact_match": round(ps["exact"], 4), "mean_jaccard": round(ps["jaccard"], 4),
                             "krippendorff_alpha": round(alpha, 4) if alpha is not None else ""})

        # per-tag kappa (worst first)
        fk = fleiss_per_tag(items, annotators)
        ranked = sorted(fk.items(), key=lambda kv: (kv[1]["kappa"] is None, kv[1]["kappa"] if kv[1]["kappa"] is not None else 0))
        print("  most-contentious codes (lowest Fleiss kappa):")
        for code, m in ranked[:8]:
            k = "n/a" if m["kappa"] is None else f"{m['kappa']:.3f}"
            print(f"      {code:8} kappa={k:>7}  support={m['support']:,}")
        tag_csv = OUTDIR / f"per_tag_kappa_phase_{ph}.csv"
        with tag_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code", "fleiss_kappa", "support_sentences", "n_sentences"])
            for code, m in sorted(fk.items()):
                w.writerow([code, "" if m["kappa"] is None else round(m["kappa"], 4), m["support"], m["n"]])

        # disagreement hotspots
        dis = disagreements(items)
        dis_csv = OUTDIR / f"disagreements_phase_{ph}.csv"
        with dis_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["debate", "unit_id", "disagreement", "text"] + [f"Annotator {a}" for a in annotators])
            for d, it in dis[:args.top]:
                w.writerow([it["file"], it["unit"], round(d, 3), it["text"]]
                           + ["; ".join(sorted(it["ratings"].get(a, frozenset()))) for a in annotators])
        print(f"  wrote {tag_csv.name}  and  {dis_csv.name}  ({len(dis):,} rows with disagreement)\n")

    if summary_rows:
        with (OUTDIR / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Summary written to {OUTDIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
