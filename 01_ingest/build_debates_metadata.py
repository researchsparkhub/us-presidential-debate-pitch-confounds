#!/usr/bin/env python3
"""
Build data/metadata/debates.csv from the inventory xlsx.

One row per (debate, candidate) with the candidate labeled winner or loser
based on whether their party won the presidential election that year.
This is the canonical metadata table the rest of the pipeline joins to.

Run:
    python 01_ingest/build_debates_metadata.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT = Path("/Users/lavanya/debate_analysis")
INVENTORY_XLSX = (
    Path("/Users/lavanya/debate_audio_project/Week_4_speaker diarization")
    / "Week_1_debate_inventory__1988_2020__with_transcript___video_search_URLs.xlsx"
)
OUT_CSV = PROJECT / "data" / "metadata" / "debates.csv"


# Party for every candidate that appears in the corpus. Used to label VP debates
# (where the contestants don't equal the election winner) and to verify the
# Presidential-debate label as well.
PARTY: dict[str, str] = {
    # Republicans
    "George H. W. Bush": "Republican",
    "George W. Bush": "Republican",
    "Bob Dole": "Republican",
    "Dick Cheney": "Republican",
    "Dan Quayle": "Republican",
    "Jack Kemp": "Republican",
    "John McCain": "Republican",
    "Sarah Palin": "Republican",
    "Mitt Romney": "Republican",
    "Paul Ryan": "Republican",
    "Donald Trump": "Republican",
    "Mike Pence": "Republican",
    "J.D. Vance": "Republican",
    # Democrats
    "Michael Dukakis": "Democratic",
    "Lloyd Bentsen": "Democratic",
    "Bill Clinton": "Democratic",
    "Al Gore": "Democratic",
    "Joe Lieberman": "Democratic",
    "John Kerry": "Democratic",
    "John Edwards": "Democratic",
    "Barack Obama": "Democratic",
    "Joe Biden": "Democratic",
    "Hillary Clinton": "Democratic",
    "Tim Kaine": "Democratic",
    "Kamala Harris": "Democratic",
    "Tim Walz": "Democratic",
    # Third party
    "Ross Perot": "Independent",
    "James Stockdale": "Independent",
}


def normalize_audio_id(raw: str) -> list[str]:
    """Some debates have split files like '1004_1, 1004_2'; emit canonical 1004."""
    if pd.isna(raw):
        return []
    raw = str(raw).strip()
    # Take first numeric token before any underscore, comma, or whitespace
    m = re.match(r"(\d{3,4})", raw)
    return [m.group(1)] if m else []


def main():
    print(f"[load] {INVENTORY_XLSX}")
    df = pd.read_excel(INVENTORY_XLSX, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "File no, and save it as .txt extension": "raw_audio_id",
        "Contestant 1": "c1",
        "Contenstant 2": "c2",
        "Contestant 3": "c3",
        "Who won?": "winner_election",
        "Party won": "party_won",
    })

    rows = []
    for _, r in df.iterrows():
        audio_ids = normalize_audio_id(r["raw_audio_id"])
        if not audio_ids:
            print(f"[skip] no audio id parsed from {r['raw_audio_id']!r}")
            continue
        debate_audio_id = audio_ids[0]

        candidates = [c for c in [r["c1"], r["c2"], r["c3"]] if pd.notna(c)]
        for cand in candidates:
            cand = str(cand).strip()
            party = PARTY.get(cand, "Unknown")
            if party == "Unknown":
                print(f"[warn] unknown party for candidate {cand!r}")

            is_winner = party == r["party_won"]
            role = "winner" if is_winner else "loser"

            rows.append({
                "debate_audio_id": debate_audio_id,
                "debate_id": r["debate_id"],
                "year": int(r["year"]),
                "debate_type": r["debate_type"],
                "debate_number": int(r["debate_number"]) if pd.notna(r["debate_number"]) else None,
                "date": pd.to_datetime(r["date"]).date().isoformat() if pd.notna(r["date"]) else "",
                "speaker": cand,
                "party": party,
                "election_winner": r["winner_election"],
                "winning_party": r["party_won"],
                "result": role,
            })

    out = pd.DataFrame(rows).sort_values(["debate_audio_id", "result"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}  ({len(out)} candidate-rows across {out['debate_audio_id'].nunique()} debates)")
    print(f"\nResult distribution:")
    print(out.groupby(["debate_type", "result"]).size().to_string())
    print(f"\nWinners by election year:")
    print(out[out["result"] == "winner"].groupby("year")["speaker"].apply(lambda s: ", ".join(sorted(set(s)))).to_string())


if __name__ == "__main__":
    main()
