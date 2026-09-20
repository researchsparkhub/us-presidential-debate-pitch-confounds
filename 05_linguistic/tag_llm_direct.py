#!/usr/bin/env python3
"""
Direct (synchronous, concurrent) fallback for the LLM DAMSL+Beads tagger.

Same prompt / schema / model as tag_llm.py, but calls the Messages API directly
with a thread pool instead of the Batch API — faster (~15-30 min) at ~2x cost.

Checkpointed & resumable: each completed chunk is appended to
outputs/llm_tags_partial.csv immediately, so a killed run (laptop sleep) loses
nothing — just re-run and it skips sentences already done. When every sentence
is tagged it assembles the final outputs/llm_tags.csv.

Requires: ANTHROPIC_API_KEY.
Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 05_linguistic/tag_llm_direct.py            # full run / resume
    python 05_linguistic/tag_llm_direct.py --workers 25
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# reuse the exact codebook / prompt / schema from the batch tagger
from tag_llm import (  # type: ignore
    SYSTEM, SCHEMA, MODEL, MAX_TOKENS, CHUNK,
    DAMSL_CODES, BEADS_CODES, OUT_CSV, load_sentences, write_prompt_md,
)

PROJECT = Path(__file__).resolve().parent.parent
PARTIAL = PROJECT / "outputs" / "llm_tags_partial.csv"

_lock = threading.Lock()


def load_done() -> set[str]:
    if not PARTIAL.exists():
        return set()
    with PARTIAL.open(newline="") as f:
        return {row["sentence_id"] for row in csv.DictReader(f)}


def tag_chunk(client, chunk):
    listing = "\n".join(f"[{sid}] {txt}" for sid, txt in chunk)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "Tag these sentences:\n" + listing}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    out = {}
    for item in json.loads(text).get("results", []):
        sid = item.get("sentence_id")
        if sid:
            dset, bset = set(item.get("damsl", [])), set(item.get("beads", []))
            out[sid] = ([c for c in DAMSL_CODES if c in dset],
                        [c for c in BEADS_CODES if c in bset])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    write_prompt_md()
    client = anthropic.Anthropic()

    sentences = load_sentences(args.limit)
    done = load_done()
    todo = [(sid, txt) for sid, txt in sentences if sid not in done]
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    print(f"[direct] {len(sentences):,} sentences, {len(done):,} already done, "
          f"{len(todo):,} to tag  →  {len(chunks):,} chunks, {args.workers} workers")

    new = not PARTIAL.exists()
    pf = PARTIAL.open("a", newline="")
    pw = csv.writer(pf)
    if new:
        pw.writerow(["sentence_id", "damsl_tags", "beads_tags", "n_damsl_tags", "n_beads_tags"])
        pf.flush()

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(tag_chunk, client, ch): ch for ch in chunks}
        for fut in as_completed(futs):
            ch = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[warn] chunk failed ({ch[0][0]}…): {e}")
                continue
            with _lock:
                for sid, txt in ch:
                    d, b = res.get(sid, ([], []))
                    pw.writerow([sid, "; ".join(d), "; ".join(b), len(d), len(b)])
                pf.flush()
            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(chunks)} chunks done")
    pf.close()

    # assemble final output from the checkpoint, in master order
    rows = {r["sentence_id"]: r for r in csv.DictReader(PARTIAL.open(newline=""))}
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "damsl_tags", "beads_tags", "n_damsl_tags", "n_beads_tags"])
        for sid, _ in sentences:
            r = rows.get(sid, {})
            w.writerow([sid, r.get("damsl_tags", ""), r.get("beads_tags", ""),
                        r.get("n_damsl_tags", "0"), r.get("n_beads_tags", "0")])
    dt = sum(1 for sid, _ in sentences if rows.get(sid, {}).get("damsl_tags"))
    bt = sum(1 for sid, _ in sentences if rows.get(sid, {}).get("beads_tags"))
    print(f"[write] {OUT_CSV}  ({len(sentences):,} rows; {dt:,} with DAMSL, {bt:,} with Beads)")


if __name__ == "__main__":
    main()
