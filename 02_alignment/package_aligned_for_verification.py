#!/usr/bin/env python3
"""
Build a self-contained verification packet for each MFA-aligned debate.

For each debate this produces ONE folder containing:
  audio/<turn_id>.wav    — sliced clip for the sampled turn
  index.html              — browser UI with audio + dropdown verdicts
  turns.csv               — structured form (audio_clip, speaker, text + empty verdicts)
  debate_info.json        — debate metadata (year, candidates, winner/loser)
  README.md               — instructions for the verifier

The verifier opens index.html in any browser, listens to clips, picks
dropdown verdicts per turn, then clicks "Download results CSV" — that's
what they send back. `merge_verification_results.py` aggregates returned
CSVs into corpus-wide accuracy metrics.

Run:
    python 02_alignment/package_aligned_for_verification.py

By default packages every debate in mfa_turns.csv. To target specific ones:
    python 02_alignment/package_aligned_for_verification.py --debates 1003 1010 1031

Output folder per debate:
    02_alignment/verification_packets/packet_<debate_id>/
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT = Path("/Users/lavanya/debate_analysis")
ALIGN_DIR = PROJECT / "02_alignment"
AUDIO_DIR = PROJECT / "data" / "audio"
TURNS_CSV = ALIGN_DIR / "mfa_turns.csv"
DEBATES_CSV = PROJECT / "data" / "metadata" / "debates.csv"
OUT_BASE = ALIGN_DIR / "verification_packets"

# Sample size per debate
N_PER_DEBATE = 20
MIN_TURN_SEC = 3.0     # skip very short interjections — boring + unhelpful
PAD_SEC = 0.20          # tiny audio pre/post-roll for listening comfort


def slice_clip(wav_path: Path, start: float, end: float) -> bytes:
    info = sf.info(str(wav_path))
    sr = info.samplerate
    s = max(0, int((start - PAD_SEC) * sr))
    e = min(info.frames, int((end + PAD_SEC) * sr))
    data, _ = sf.read(str(wav_path), start=s, stop=e, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def stratified_sample(turns: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n turns spanning the debate timeline. Eligible = duration ≥ MIN_TURN_SEC."""
    eligible = turns[turns["duration_sec"] >= MIN_TURN_SEC].copy()
    if len(eligible) <= n:
        return eligible.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    # Three time strata of roughly equal size
    n_each = n // 3
    n_extra = n - 3 * n_each
    end_t = eligible["end_sec"].max()
    early = eligible[eligible["start_sec"] <= end_t / 3]
    mid = eligible[(eligible["start_sec"] > end_t / 3) & (eligible["start_sec"] <= 2 * end_t / 3)]
    late = eligible[eligible["start_sec"] > 2 * end_t / 3]

    parts = []
    for sub, k in [(early, n_each + n_extra), (mid, n_each), (late, n_each)]:
        if len(sub) == 0:
            continue
        take = min(k, len(sub))
        parts.append(sub.sample(n=take, random_state=int(rng.integers(0, 2**31))))

    out = pd.concat(parts, ignore_index=True).sort_values("start_sec").reset_index(drop=True)
    return out


def fmt_time(t: float) -> str:
    m = int(t // 60); s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Alignment verification — Debate {debate_id} ({year}, {debate_type})</title>
<style>
:root {{ --good: #1b873f; --warn: #c47b00; --bad: #b3261e; --bg: #fafafa; --line: #ddd; }}
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; color: #222; background: white; }}
h1 {{ margin-bottom: 0.2em; }}
.meta {{ color: #555; margin-bottom: 1em; }}
.toolbar {{ position: sticky; top: 0; background: white; padding: 12px 0; border-bottom: 1px solid var(--line); z-index: 10; }}
.tally {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 14px; margin-top: 8px; }}
.tally span {{ background: #eef; padding: 4px 10px; border-radius: 12px; }}
.tally .green {{ background: #e6f4ea; color: var(--good); }}
.tally .red {{ background: #fde9e7; color: var(--bad); }}
.tally .yellow {{ background: #fff4d6; color: var(--warn); }}
.btn {{ padding: 7px 14px; border-radius: 4px; border: 1px solid #888; background: white; cursor: pointer; font-size: 14px; }}
.btn-primary {{ background: #1976d2; color: white; border-color: #1976d2; }}
.btn-danger {{ background: white; color: var(--bad); border-color: var(--bad); }}
.turn {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin: 14px 0; background: var(--bg); display: grid; grid-template-columns: 1.1fr 2fr 2.5fr; gap: 16px; align-items: start; }}
.turn .meta-col b {{ display: block; font-size: 15px; }}
.turn .meta-col .role {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-top: 4px; }}
.role.winner {{ background: #e6f4ea; color: var(--good); }}
.role.loser {{ background: #fde9e7; color: var(--bad); }}
.role.moderator {{ background: #eee; color: #555; }}
.turn .meta-col small {{ color: #666; display: block; margin-top: 4px; }}
.turn .audio-col audio {{ width: 100%; }}
.turn .verdict-col {{ display: flex; flex-direction: column; gap: 8px; }}
.turn .verdict-col label {{ display: grid; grid-template-columns: 140px 1fr; align-items: center; gap: 10px; font-size: 13px; }}
.turn .verdict-col select, .turn .verdict-col input {{ padding: 5px 8px; font-size: 13px; }}
.text {{ font-style: italic; line-height: 1.4; max-height: 160px; overflow-y: auto; padding: 6px 8px; background: white; border: 1px solid #e5e5e5; border-radius: 4px; }}
.text mark {{ background: #fff48a; padding: 0 1px; }}
small.dim {{ color: #888; }}
</style>
</head>
<body>

<h1>Alignment verification — Debate {debate_id}</h1>
<div class="meta">
  {year} · {debate_type} · {date} · {candidates} ·
  <b>Winner ticket:</b> {winning_party}
</div>

<div class="toolbar">
  <button class="btn btn-primary" onclick="downloadCSV()">💾 Download results CSV</button>
  <button class="btn" onclick="loadDraft()">↩︎ Reload draft</button>
  <button class="btn btn-danger" onclick="if(confirm('Clear all your verdicts?')) clearAll();">🗑 Clear all</button>
  <span style="margin-left:1em;"><small class="dim">Your work auto-saves to your browser as you go.</small></span>
  <div class="tally" id="tally"></div>
</div>

<p>For each of the {n_turns} sampled turns below, listen to the audio clip and judge:</p>
<ol>
  <li><b>Audio matches text</b> — does the spoken content match the displayed transcript text?</li>
  <li><b>Alignment timing</b> — does the clip start near the first word and end near the last word of the turn?</li>
  <li><b>Speaker correct</b> — is the labeled speaker the person you actually hear?</li>
</ol>
<p>When you're done with all {n_turns}, click <b>💾 Download results CSV</b> at the top and email the file back.</p>

<div id="turns">
{turns_html}
</div>

<script>
const PACKET_ID = "{packet_id}";
const STORAGE_KEY = "dap_verify_" + PACKET_ID;

const FIELDS = ["audio_matches_text", "alignment_timing", "speaker_correct", "notes"];

function saveDraft() {{
  const data = {{}};
  document.querySelectorAll(".turn").forEach(t => {{
    const tid = t.dataset.turn_id;
    data[tid] = {{}};
    FIELDS.forEach(f => {{
      const el = t.querySelector(`[data-field="${{f}}"]`);
      if (el) data[tid][f] = el.value;
    }});
  }});
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  updateTally();
}}

function loadDraft() {{
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {{
    const data = JSON.parse(raw);
    Object.entries(data).forEach(([tid, vals]) => {{
      const t = document.querySelector(`.turn[data-turn_id="${{tid}}"]`);
      if (!t) return;
      FIELDS.forEach(f => {{
        const el = t.querySelector(`[data-field="${{f}}"]`);
        if (el && vals[f] !== undefined) el.value = vals[f];
      }});
    }});
    updateTally();
  }} catch (e) {{}}
}}

function clearAll() {{
  document.querySelectorAll(".turn select, .turn input").forEach(el => el.value = "");
  saveDraft();
}}

function updateTally() {{
  let total = document.querySelectorAll(".turn").length;
  let nText=0, nTextOK=0, nTextWrong=0;
  let nTime=0, nTimeOK=0, nTimeOff=0;
  let nSpk=0,  nSpkOK=0,  nSpkWrong=0;
  let reviewed = 0;

  document.querySelectorAll(".turn").forEach(t => {{
    const text = t.querySelector(`[data-field="audio_matches_text"]`).value;
    const time = t.querySelector(`[data-field="alignment_timing"]`).value;
    const spk  = t.querySelector(`[data-field="speaker_correct"]`).value;

    if (text) {{ nText++; if (text==="yes") nTextOK++; else if (text==="no") nTextWrong++; }}
    if (time) {{ nTime++; if (time==="precise") nTimeOK++; else if (time==="off") nTimeOff++; }}
    if (spk)  {{ nSpk++;  if (spk==="yes") nSpkOK++; else if (spk==="no") nSpkWrong++; }}
    if (text && time && spk) reviewed++;
  }});

  document.getElementById("tally").innerHTML =
    `<span>📊 reviewed: <b>${{reviewed}}/${{total}}</b></span>` +
    `<span class="green">📄 text ok: ${{nTextOK}}/${{nText}}</span>` +
    `<span class="red">📄 text wrong: ${{nTextWrong}}/${{nText}}</span>` +
    `<span class="green">⏱ time precise: ${{nTimeOK}}/${{nTime}}</span>` +
    `<span class="red">⏱ time off: ${{nTimeOff}}/${{nTime}}</span>` +
    `<span class="green">👤 speaker ok: ${{nSpkOK}}/${{nSpk}}</span>` +
    `<span class="red">👤 speaker wrong: ${{nSpkWrong}}/${{nSpk}}</span>`;
}}

function csvEscape(v) {{
  if (v == null) return "";
  v = String(v);
  if (/[,\"\n]/.test(v)) return '"' + v.replace(/"/g, '""') + '"';
  return v;
}}

function downloadCSV() {{
  const headers = [
    "packet_id", "debate_audio_id", "turn_id", "speaker_raw", "start_sec", "end_sec",
    "duration_sec", "n_words", "text",
    "audio_matches_text", "alignment_timing", "speaker_correct", "notes",
  ];
  const lines = [headers.join(",")];
  document.querySelectorAll(".turn").forEach(t => {{
    const row = headers.map(h => {{
      if (t.dataset[h] !== undefined) return csvEscape(t.dataset[h]);
      const el = t.querySelector(`[data-field="${{h}}"]`);
      return el ? csvEscape(el.value) : "";
    }});
    lines.push(row.join(","));
  }});
  const blob = new Blob([lines.join("\n")], {{ type: "text/csv" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = PACKET_ID + "_verified.csv";
  a.click();
}}

document.addEventListener("change", saveDraft);
document.addEventListener("input", saveDraft);
window.addEventListener("load", () => {{ loadDraft(); updateTally(); }});
</script>
</body>
</html>
"""


def render_turn_html(row: dict, speakers_for_dropdown: list[str]) -> str:
    speaker_opts = "".join(f'<option value="{s}">{s}</option>' for s in speakers_for_dropdown)
    text = str(row["text"]).replace("\n", " ").replace("\"", "&quot;")
    role_class = ""
    role_label = row.get("role_label", "")
    if role_label == "winner":
        role_class = "winner"
    elif role_label == "loser":
        role_class = "loser"
    elif role_label == "moderator":
        role_class = "moderator"

    return f"""
<div class="turn" data-turn_id="{row['turn_id']}"
     data-packet_id="{row['packet_id']}"
     data-debate_audio_id="{row['debate_audio_id']}"
     data-speaker_raw="{row['speaker_raw']}"
     data-start_sec="{row['start_sec']:.3f}"
     data-end_sec="{row['end_sec']:.3f}"
     data-duration_sec="{row['duration_sec']:.2f}"
     data-n_words="{row['n_words']}"
     data-text="{text}">
  <div class="meta-col">
    <b>{row['speaker_raw']}</b>
    {f'<span class="role {role_class}">{role_label}</span>' if role_label else ''}
    <small>{fmt_time(row['start_sec'])} → {fmt_time(row['end_sec'])}</small>
    <small>{row['duration_sec']:.1f}s · {row['n_words']} words</small>
  </div>
  <div class="audio-col">
    <audio controls preload="none" src="audio/{row['turn_id']}.wav"></audio>
    <div class="text" style="margin-top:8px;">{text}</div>
  </div>
  <div class="verdict-col">
    <label>📄 Audio matches text?
      <select data-field="audio_matches_text">
        <option value=""></option>
        <option value="yes">yes — verbatim</option>
        <option value="partial">partial — most matches</option>
        <option value="no">no — different content</option>
        <option value="unclear">unclear — can't tell</option>
      </select>
    </label>
    <label>⏱ Alignment timing?
      <select data-field="alignment_timing">
        <option value=""></option>
        <option value="precise">precise — within 1s</option>
        <option value="approximate">approximate — 1-5s off</option>
        <option value="off">off — &gt;5s wrong</option>
        <option value="unclear">unclear</option>
      </select>
    </label>
    <label>👤 Speaker correct?
      <select data-field="speaker_correct">
        <option value=""></option>
        <option value="yes">yes</option>
        <option value="no">no — different speaker</option>
        <option value="unclear">unclear</option>
      </select>
    </label>
    <label>📝 Notes
      <input type="text" data-field="notes" placeholder="optional…">
    </label>
  </div>
</div>
"""


def build_packet(did: str, turns_df: pd.DataFrame, debates_df: pd.DataFrame, out_dir: Path):
    out = out_dir / f"packet_{did}"
    audio_out = out / "audio"
    out.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(exist_ok=True)

    turns_did = turns_df[turns_df["debate_audio_id"].astype(str) == str(did)].copy()
    if turns_did.empty:
        print(f"[skip] {did}: no turns in mfa_turns.csv")
        return None

    sampled = stratified_sample(turns_did, N_PER_DEBATE, seed=abs(hash(("verify", did))) % (2**31))
    if sampled.empty:
        print(f"[skip] {did}: no eligible turns after duration filter")
        return None

    # Debate metadata
    info_rows = debates_df[debates_df["debate_audio_id"].astype(str) == str(did)]
    candidates = [f"{r['speaker']} ({r['result']})" for _, r in info_rows.iterrows()]
    info = {
        "debate_audio_id": did,
        "debate_id": str(info_rows.iloc[0]["debate_id"]) if not info_rows.empty else "",
        "year": int(info_rows.iloc[0]["year"]) if not info_rows.empty else 0,
        "debate_type": str(info_rows.iloc[0]["debate_type"]) if not info_rows.empty else "",
        "date": str(info_rows.iloc[0]["date"]) if not info_rows.empty else "",
        "candidates": candidates,
        "election_winner": str(info_rows.iloc[0]["election_winner"]) if not info_rows.empty else "",
        "winning_party": str(info_rows.iloc[0]["winning_party"]) if not info_rows.empty else "",
        "n_sampled_turns": len(sampled),
    }
    (out / "debate_info.json").write_text(json.dumps(info, indent=2))

    # Build a speaker→role map for color-coding in the UI
    role_map: dict[str, str] = {}
    for _, r in info_rows.iterrows():
        # Match speakers by last-name first heuristic (raw labels are uppercase last names typically)
        last_name = str(r["speaker"]).split()[-1].upper()
        role_map[last_name] = r["result"]
    # Generic moderator/panelist heuristic
    moderator_indicators = {"LEHRER", "JENNINGS", "SHAW", "SIMPSON", "RUSSERT", "MASHEK", "GROER",
                             "BURNS", "DAVID", "KITE", "FLECK", "SCHIEFFER", "BLITZER", "RADDATZ",
                             "TAPPER", "BASH", "DAVIS", "MUIR", "BERKLEY", "MITCHELL", "WARNER",
                             "COMPTON", "BROKAW", "BREWER", "HARWOOD", "STEPHANOPOULOS"}

    # Slice audio and accumulate rows for both CSV and HTML
    wav_path = AUDIO_DIR / f"{did}.wav"
    if not wav_path.exists():
        print(f"[skip] {did}: WAV missing at {wav_path}")
        return None

    csv_rows: list[dict] = []
    html_parts: list[str] = []
    n_clips_written = 0
    for _, r in sampled.iterrows():
        spk_raw = str(r["speaker_raw"]).strip()
        # Determine role label for this turn
        spk_token = spk_raw.split()[-1].upper() if spk_raw else ""
        if spk_token in role_map:
            role_label = role_map[spk_token]
        elif spk_token in moderator_indicators:
            role_label = "moderator"
        elif spk_token.startswith(("MS.", "MR.", "DR.")):
            role_label = "audience"
        else:
            role_label = ""

        try:
            clip = slice_clip(wav_path, float(r["start_sec"]), float(r["end_sec"]))
        except Exception as e:
            print(f"[warn] {did} {r['turn_id']}: clip slice failed: {e}")
            continue
        (audio_out / f"{r['turn_id']}.wav").write_bytes(clip)
        n_clips_written += 1

        row_dict = {
            "packet_id": f"verify_{did}",
            "debate_audio_id": did,
            "turn_id": r["turn_id"],
            "speaker_raw": spk_raw,
            "role_label": role_label,
            "start_sec": float(r["start_sec"]),
            "end_sec": float(r["end_sec"]),
            "duration_sec": float(r["duration_sec"]),
            "n_words": int(r["n_words"]),
            "text": str(r["text"]),
        }
        csv_rows.append(row_dict)
        html_parts.append(render_turn_html(row_dict, []))

    # Save the structured CSV (with empty verdict columns)
    csv_path = out / "turns.csv"
    df_out = pd.DataFrame(csv_rows)
    for col in ("audio_matches_text", "alignment_timing", "speaker_correct", "notes"):
        df_out[col] = ""
    df_out.to_csv(csv_path, index=False)

    # Render the HTML
    html = HTML_TEMPLATE.format(
        debate_id=info["debate_id"],
        year=info["year"],
        debate_type=info["debate_type"],
        date=info["date"],
        candidates=", ".join(candidates),
        winning_party=info["winning_party"],
        n_turns=len(csv_rows),
        packet_id=f"verify_{did}",
        turns_html="\n".join(html_parts),
    )
    (out / "index.html").write_text(html, encoding="utf-8")

    # README
    readme = f"""# Alignment verification packet — Debate {did}

**Debate:** {info['year']} {info['debate_type']} ({info['date']})
**Candidates:** {", ".join(candidates)}
**Election winner:** {info['election_winner']} ({info['winning_party']} party)

This packet contains {n_clips_written} sampled turns from the MFA-aligned debate.

## How to verify (no install needed)

1. Open `index.html` in any modern browser (Chrome, Safari, Firefox).
2. For each turn:
   - Listen to the audio clip and read the transcript text shown next to it.
   - Pick a verdict for each of the three questions on the right.
   - Your draft auto-saves to your browser as you go.
3. When you're done with all {n_clips_written}, click the **💾 Download results CSV**
   button at the top — that file is what you send back.

## What the verdicts mean

- **Audio matches text** — does the spoken audio match the transcript text shown?
  - `yes — verbatim`: every word matches (small contractions like "I'll" / "I will" OK)
  - `partial — most matches`: the gist is the same but some wording differs
  - `no — different content`: the audio is clearly saying something different
  - `unclear`: audio quality / accent makes it hard to tell

- **Alignment timing** — does the clip start at the first word and end at the last word?
  - `precise — within 1s`: clip starts/ends right at the spoken content
  - `approximate — 1-5s off`: small mismatch but the right region
  - `off — >5s wrong`: clip is in the wrong part of the debate
  - `unclear`

- **Speaker correct** — is the speaker labeled in the metadata box the person you actually hear?
  - `yes` / `no — different speaker` / `unclear`

## Sampling

20 turns randomly sampled per debate, stratified across the timeline
(early / middle / late thirds) with a 3-second minimum duration filter
(so you're not asked to judge 1-word interjections).
"""
    (out / "README.md").write_text(readme)

    print(f"[done] {did}: {n_clips_written} turns packaged → {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-csv", default=str(TURNS_CSV))
    ap.add_argument("--debates-csv", default=str(DEBATES_CSV))
    ap.add_argument("--out-dir", default=str(OUT_BASE))
    ap.add_argument("--debates", nargs="*", help="Specific debate_audio_ids (default: all in mfa_turns)")
    ap.add_argument("--zip", action="store_true", help="Also zip each packet folder")
    args = ap.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    turns_df = pd.read_csv(args.turns_csv)
    turns_df["debate_audio_id"] = turns_df["debate_audio_id"].astype(str)
    debates_df = pd.read_csv(args.debates_csv)

    target = args.debates if args.debates else sorted(turns_df["debate_audio_id"].unique())
    print(f"Packaging {len(target)} debates into {out_base}/\n")

    built = []
    for did in target:
        out = build_packet(str(did), turns_df, debates_df, out_base)
        if out:
            built.append(out)

    print(f"\nBuilt {len(built)} packets.")

    if args.zip:
        print("\nZipping packets…")
        for out in built:
            shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
            print(f"  {out.name}.zip")

    print("\nNext steps:")
    print(f"  - Send each packet folder (or .zip) to your verifier.")
    print(f"  - When they return the *_verified.csv files, place them in:")
    print(f"      {out_base}/returned/")
    print(f"  - Then run: python merge_verification_results.py")


if __name__ == "__main__":
    main()
