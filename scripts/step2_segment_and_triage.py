#!/usr/bin/env python3
"""Build bidirectional turn candidates from the Step 1 manifest without decoding video."""

import csv
import json
import math
import random
import re
import tempfile
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
EPS = 1e-9
BACKCHANNEL_MAX = 0.6
SNAP_MAX = 0.25
DBFS_FLOOR = -120.0
STATUSES = ("pass", "review", "reject")
BUCKETS = ("short", "core", "long")
SAMPLE_SEED = 20261120
AUDIT_ONLY = ("prev_other_turn_id", "prev_other_text", "prev_other_gap_sec")
GROUP_KEYS = (
    "conversation_id", "vendor_id", "session_id", "interaction_id", "prompt_hash", "prompt_id_unique",
    "participant_a", "participant_b", "official_vendor_id", "official_session_id",
    "official_participant_a", "official_participant_b",
)
WORDS = re.compile(r"[^\W_]+(?:[’'\-‐‑][^\W_]+)*", re.UNICODE)
END_QUOTES = "\"'’”»)]}"
FILLERS = {
    "uh", "um", "uh-huh", "mhm", "mm-hmm", "mm-hm", "hmm-mm", "mm", "hm", "hmm",
    "yeah", "yep", "ok", "okay", "right", "oh",
}
FILLER_PAIRS = {("uh", "huh"), ("mm", "hmm"), ("mm", "hm"), ("m", "hm"), ("m", "m")}


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    word_fallbacks: int = 0


@dataclass
class Participant:
    participant_id: str
    audio: np.ndarray
    sample_rate: int
    vad: list
    transcript: list
    box_valid: np.ndarray
    move_valid: np.ndarray

    @property
    def duration(self):
        return len(self.audio) / self.sample_rate


def terminal_punct(text):
    return text.rstrip().rstrip(END_QUOTES).rstrip().endswith((".", "?", "!"))


def valid_span(start, end):
    return (isinstance(start, (int, float)) and isinstance(end, (int, float))
            and math.isfinite(start) and math.isfinite(end) and start <= end)


def load_transcript(entries):
    segments = []
    for entry in entries:
        start, end = entry.get("start"), entry.get("end")
        if not valid_span(start, end):
            raise ValueError(f"Invalid transcript segment bounds: {start!r}, {end!r}")
        word_spans, word_texts = [], []
        fallbacks = 0
        for word in entry.get("words", []):
            token = word["word"].strip()
            word_texts.append(token)
            lo, hi = word.get("start"), word.get("end")
            if not valid_span(lo, hi):
                lo, hi = start, end
                fallbacks += 1
            word_spans.append((lo, hi))
        # Keep the original segment envelope, widened only if valid word bounds extend it.
        # Use the original segment text including untimed tokens and punctuation.
        text = (entry.get("transcript") or "").strip() or " ".join(word_texts).strip()
        segments.append(Segment(min([start, *(lo for lo, _ in word_spans)]),
                                max([end, *(hi for _, hi in word_spans)]), text, fallbacks))
    return sorted(segments, key=lambda s: (s.start, s.end))


def union_intervals(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1] + EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def read_vad(entries, duration):
    spans = []
    for entry in entries:
        lo, hi = entry.get("start"), entry.get("end")
        if not valid_span(lo, hi):
            raise ValueError(f"Invalid VAD bounds: {lo!r}, {hi!r}")
        spans.append((max(0.0, lo), min(duration, hi)))
    return union_intervals(spans)


def load_participant(stem, pid):
    with stem.with_suffix(".wav").open("rb") as source, sf.SoundFile(source, mode="r") as wav:
        if wav.channels != 1 or wav.subtype != "FLOAT" or wav.samplerate <= 0:
            raise ValueError(f"Expected mono float WAV: {stem}.wav")
        rate = wav.samplerate
        samples = wav.read(dtype="float32")
    if not samples.size:
        raise ValueError(f"Empty audio: {stem}.wav")
    with stem.with_suffix(".json").open(encoding="utf-8") as source:
        annotations = json.load(source)
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        boxes = archive["boxes_and_keypoints:is_valid_box"]
        moves = archive["movement:is_valid"]
        if boxes.ndim != 1 or moves.ndim not in (1, 2) or (moves.ndim == 2 and moves.shape[1] != 1):
            raise ValueError(f"Unexpected NPZ mask shapes in {stem}.npz: {boxes.shape}, {moves.shape}")
        boxes = np.isfinite(boxes) & (boxes > 0.5)
        moves = moves.reshape(-1)
        moves = np.isfinite(moves) & (moves > 0.5)
    return Participant(pid, samples, rate, read_vad(annotations["metadata:vad"], len(samples) / rate),
                       load_transcript(annotations["metadata:transcript"]), boxes, moves)


def overlap(lo, hi, start, end):
    return max(0.0, min(hi, end) - max(lo, start))


def listener_takes_floor(vad, gap_start, gap_end):
    # A sustained VAD that intersects the gap counts even if only part of it falls
    # inside the gap. This catches a takeover that starts late and spills over.
    return any(hi - lo > BACKCHANNEL_MAX + EPS and overlap(lo, hi, gap_start, gap_end) > EPS
               for lo, hi in vad)


def merge_segments(segments, listener_vad):
    merged = []
    for segment in segments:
        if not merged:
            merged.append([segment])
            continue
        prev = merged[-1]
        prev_end = max(s.end for s in prev)
        gap = segment.start - prev_end
        close = gap <= 0.8 + EPS or (gap <= 1.2 + EPS and not terminal_punct(prev[-1].text))
        if close and not listener_takes_floor(listener_vad, prev_end, segment.start):
            prev.append(segment)
        else:
            merged.append([segment])
    return merged


def clip_time(t, duration):
    return max(0.0, min(duration, t))


def turn_boundaries(groups, speaker_vad, duration):
    bounds = [(min(s.start for s in group), max(s.end for s in group)) for group in groups]
    prev_end = 0.0
    for i, (raw_start, raw_end) in enumerate(bounds):
        matched = [(lo, hi) for lo, hi in speaker_vad if overlap(lo, hi, raw_start, raw_end) > EPS]
        left = [lo for lo, _ in matched if 0 < raw_start - lo <= SNAP_MAX + EPS]
        right = [hi for _, hi in matched if 0 < hi - raw_end <= SNAP_MAX + EPS]
        start = max(prev_end, clip_time(min([raw_start, *left]), duration))
        end = clip_time(max([raw_end, *right]), duration)
        if i + 1 < len(bounds):
            end = min(end, clip_time(bounds[i + 1][0], duration))
        end = max(start, end)
        yield raw_start, raw_end, start, end
        prev_end = end


def transcript_vad_ratio(group, vad, duration):
    raw = union_intervals((clip_time(s.start, duration), clip_time(s.end, duration)) for s in group)
    denom = sum(hi - lo for lo, hi in raw)
    return min(1.0, sum(overlap(lo, hi, s, e) for lo, hi in vad for s, e in raw) / denom) if denom else 0.0


def overlap_metrics(vad, start, end):
    short = substantive = 0.0
    late = False
    for lo, hi in vad:
        occupied = overlap(lo, hi, start, end)
        if occupied <= EPS:
            continue
        if hi - lo <= BACKCHANNEL_MAX + EPS and hi <= end + EPS:
            short += occupied
        else:
            substantive += occupied
            late |= overlap(lo, hi, max(start, end - 0.5), end) > EPS
    return short, substantive, late


def audio_index(t, rate, length):
    # Half-open windows on the sample clock: include samples at start but not at end.
    return min(length, max(0, math.ceil(t * rate - 1e-7)))


def power_stats(audio, start, end):
    window = audio[start:end]
    return float(np.einsum("i,i->", window, window, dtype=np.float64)), window.size


def power_dbfs(energy, count):
    if not math.isfinite(energy):
        raise ValueError("Nonfinite samples in an audio analysis window")
    return max(DBFS_FLOOR, 10 * math.log10(energy / count)) if energy > 0 and count else DBFS_FLOOR


def speaker_dbfs(person, start, end):
    index = lambda t: audio_index(t, person.sample_rate, len(person.audio))
    return power_dbfs(*power_stats(person.audio, index(start), index(end)))


def listener_silent_dbfs(person, start, end):
    index = lambda t: audio_index(t, person.sample_rate, len(person.audio))
    pos, stop = index(start), index(end)
    energy, count = 0.0, 0
    for lo, hi in person.vad:
        if hi <= start or lo >= end:
            continue
        cut_start, cut_end = index(max(lo, start)), index(min(hi, end))
        if pos < cut_start:
            e, n = power_stats(person.audio, pos, cut_start)
            energy, count = energy + e, count + n
        pos = max(pos, cut_end)
    if pos < stop:
        e, n = power_stats(person.audio, pos, stop)
        energy, count = energy + e, count + n
    return power_dbfs(energy, count) if count else None


def mask_metrics(mask, start, end):
    if end <= start:
        return 0.0, 0, 0
    first = max(0, min(len(mask), math.floor(start * FPS + EPS)))
    last = max(first, min(len(mask), math.ceil(end * FPS - EPS)))
    window = mask[first:last]
    if not window.size:
        return 0.0, 0, 0
    changes = np.flatnonzero(np.diff(np.r_[False, ~window, False]))
    return float(window.mean()), int(np.max(changes[1::2] - changes[::2], initial=0)), len(window)


def feature_metrics(prefix, person, start, end):
    box, box_run, box_n = mask_metrics(person.box_valid, start, end)
    move, move_run, move_n = mask_metrics(person.move_valid, start, end)
    return {
        f"{prefix}_box_valid_ratio": box, f"{prefix}_box_max_consec_invalid": box_run,
        f"{prefix}_box_frame_count": box_n,
        f"{prefix}_move_valid_ratio": move, f"{prefix}_move_max_consec_invalid": move_run,
        f"{prefix}_move_frame_count": move_n,
    }


def minimal_filler(words, duration):
    normalized = tuple(w.casefold().replace("‐", "-").replace("‑", "-") for w in words)
    return (duration < 1.2 - EPS and 1 <= len(normalized) <= 2
            and (all(w in FILLERS for w in normalized) or normalized in FILLER_PAIRS))


def triage(row):
    reject, review = [], []
    ratio = row["substantive_overlap_ratio"]
    if not row["text"].strip():
        reject.append("EMPTY_TEXT")
    if row["duration_sec"] < 0.4 - EPS:
        reject.append("DURATION_LT_0_4S")
    if row["is_minimal_filler"]:
        reject.append("MINIMAL_FILLER")
    if ratio > 0.40 + EPS:
        reject.append("SEVERE_SUBSTANTIVE_OVERLAP")
    if row["speaker_rms_db"] < -55.0:
        reject.append("SPEAKER_NEAR_SILENT")
    for side in ("speaker", "listener"):
        if row[f"{side}_move_valid_ratio"] < 0.90:
            review.append(f"{side.upper()}_MOVE_VALID_RATIO_LOW")
        if row[f"{side}_move_max_consec_invalid"] >= 5:
            review.append(f"{side.upper()}_MOVE_INVALID_RUN")
        if row[f"{side}_box_valid_ratio"] < 0.90:
            review.append(f"{side.upper()}_BOX_VALID_RATIO_LOW")
    if 0.15 + EPS < ratio <= 0.40 + EPS:
        review.append("MODERATE_SUBSTANTIVE_OVERLAP")
    if row["late_overlap_flag"]:
        review.append("LATE_SUBSTANTIVE_OVERLAP")
    if row["transcript_vad_overlap_ratio"] < 0.50 - EPS:
        review.append("LOW_TRANSCRIPT_VAD_OVERLAP")
    if row["crosstalk_snr_db"] is not None and row["crosstalk_snr_db"] < 8.0:
        review.append("LOW_CROSSTALK_SNR")
    if not row["has_terminal_punct"]:
        review.append("NO_TERMINAL_PUNCT")
    return ("reject" if reject else "review" if review else "pass"), reject, review


def build_turns(meta, sr, lr, speaker, listener, duration):
    groups = merge_segments(speaker.transcript, listener.vad)
    for group, (raw_start, raw_end, start, end) in zip(groups, turn_boundaries(groups, speaker.vad, duration)):
        length = end - start
        bucket = "short" if length < 2.0 - EPS else "long" if length > 12.0 + EPS else "core"
        text = " ".join(segment.text for segment in group if segment.text)
        words = WORDS.findall(text)
        ref_start, ref_end = max(0.0, end - 0.5), min(duration, end + 2.0)
        active_ref = sum(overlap(lo, hi, ref_start, ref_end) for lo, hi in listener.vad)
        backchannel_sec, substantive_sec, late = overlap_metrics(listener.vad, start, end)
        speaker_db = speaker_dbfs(speaker, start, end)
        silent_db = listener_silent_dbfs(listener, start, end)
        row = {
            **{key: meta[key] for key in GROUP_KEYS},
            "speaker_id": speaker.participant_id, "listener_id": listener.participant_id,
            "speaker_participant_id": speaker.participant_id, "listener_participant_id": listener.participant_id,
            "official_speaker_participant_id": meta[f"official_participant_{sr}"],
            "official_listener_participant_id": meta[f"official_participant_{lr}"],
            "speaker_role": sr.upper(), "listener_role": lr.upper(), "text": text,
            "raw_transcript_start": raw_start, "raw_transcript_end": raw_end,
            "t_start": start, "t_end": end, "time_axis_end_sec": duration,
            "ref_b_start": ref_start, "ref_b_end": ref_end,
            "ref_b_start_clipped": end - 0.5 < -EPS, "ref_b_end_clipped": end + 2.0 > duration + EPS,
            "duration_sec": length, "duration_bucket": bucket, "word_count": len(words),
            "segment_count": len(group), "word_timestamp_fallback_count": sum(s.word_fallbacks for s in group),
            "has_terminal_punct": terminal_punct(text), "is_minimal_filler": minimal_filler(words, length),
            "transcript_vad_overlap_ratio": transcript_vad_ratio(group, speaker.vad, duration),
            "backchannel_overlap_sec": backchannel_sec, "substantive_overlap_sec": substantive_sec,
            "substantive_overlap_ratio": min(1.0, substantive_sec / length) if length else 0.0,
            "late_overlap_flag": late, "speaker_rms_db": speaker_db, "listener_silent_rms_db": silent_db,
            "crosstalk_snr_db": speaker_db - silent_db if silent_db is not None else None,
            "listener_ref_vad_ratio": min(1.0, active_ref / (ref_end - ref_start)) if ref_end > ref_start else 0.0,
            **feature_metrics("speaker", speaker, start, end),
            **feature_metrics("listener", listener, ref_start, ref_end),
        }
        status, reject, review = triage(row)
        row.update(triage_status=status, reason_codes=reject + review,
                   reject_reason_codes=reject, review_reason_codes=review)
        yield row


def attach_previous_turns(rows):
    by_role = {role: sorted((r for r in rows if r["speaker_role"] == role), key=lambda r: r["t_end"])
               for role in ("A", "B")}
    ends = {role: [r["t_end"] for r in turns] for role, turns in by_role.items()}
    for row in rows:
        other = row["listener_role"]
        idx = bisect_right(ends[other], row["t_start"] + EPS) - 1
        prev = by_role[other][idx] if idx >= 0 else None
        row.update(prev_other_turn_id=prev["turn_id"] if prev else None,
                   prev_other_text=prev["text"] if prev else None,
                   prev_other_gap_sec=max(0.0, row["t_start"] - prev["t_end"]) if prev else None)


def atomic_output(path, write, binary=False):
    kwargs = {"mode": "w+b"} if binary else {"mode": "w", "encoding": "utf-8", "newline": ""}
    tmp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, **kwargs)
    staging = Path(tmp.name)
    try:
        with tmp:
            write(tmp)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def save_outputs(metadata, rows, report):
    items = [{"turn_id": r["turn_id"], **r} for r in rows]
    fields = tuple(items[0])
    list_fields = ("reason_codes", "reject_reason_codes", "review_reason_codes")

    def write_csv(file):
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, **{key: json.dumps(row[key]) for key in list_fields}} for row in items)

    table = pa.Table.from_pylist(items)
    table = table.replace_schema_metadata({b"audit_only_columns": json.dumps(AUDIT_ONLY).encode(),
                                           b"time_axis": b"seconds on standalone WAV sample clock"})
    atomic_output(metadata / "turns_candidates.parquet", lambda f: pq.write_table(table, f), binary=True)
    atomic_output(metadata / "turns_candidates.csv", write_csv)
    atomic_output(metadata / "step2_triage_report.json",
                  lambda f: json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False))


def summarize(rows, session_count, no_transcript_count):
    status = Counter(r["triage_status"] for r in rows)
    buckets = Counter((r["triage_status"], r["duration_bucket"]) for r in rows)
    reasons = {s: Counter(code for r in rows if r["triage_status"] == s for code in r["reason_codes"])
               for s in STATUSES}
    decisions = {s: Counter(code for r in rows if r["triage_status"] == s for code in r[f"{s}_reason_codes"])
                 for s in ("reject", "review")}
    passed = [r for r in rows if r["triage_status"] == "pass"]
    sample = random.Random(SAMPLE_SEED).sample(passed, min(5, len(passed)))
    sample_fields = ("turn_id", "duration_sec", "text", "substantive_overlap_ratio")
    return {
        "sessions_processed": session_count,
        "directions_processed": 2 * session_count,
        "directions_without_transcript_segments": no_transcript_count,
        "transcript_segments": sum(r["segment_count"] for r in rows),
        "word_timestamp_fallbacks": sum(r["word_timestamp_fallback_count"] for r in rows),
        "total_candidates": len(rows),
        "status_counts": {s: status[s] for s in STATUSES},
        "status_percentages": {s: 100 * status[s] / len(rows) for s in STATUSES},
        "duration_bucket_by_status": {s: {b: buckets[s, b] for b in BUCKETS} for s in STATUSES},
        "reason_counts_by_status": {s: dict(reasons[s].most_common()) for s in STATUSES},
        "decision_reason_counts_by_status": {s: dict(c.most_common()) for s, c in decisions.items()},
        "pass_sample_seed": SAMPLE_SEED,
        "pass_samples": [{k: row[k] for k in sample_fields} for row in sample],
        "audit_only_columns": list(AUDIT_ONLY),
        "notes": {
            "time_axis": "Seconds on the standalone WAV sample clock, capped by the shorter of the two WAVs.",
            "gap_takeover": "A listener VAD run longer than 0.6 s with a positive intersection with the inter-segment gap blocks either merge rule.",
            "transcript_vad_overlap": "Own VAD coverage of the union of raw segment spans clipped to the common WAV timeline; excludes inter-segment gaps.",
            "reference_clipping": "Flags indicate the nominal reference window would exceed the common media window [0, T].",
            "audio": "RMS floor is -120 dBFS. Listener RMS excludes own VAD; no remaining samples gives null RMS and SNR.",
            "npz": "30 fps; count every frame intersecting the window before clamping to array bounds. Empty slices have ratio and longest run set to zero; frame count is saved.",
            "reason_codes": "All triggered codes are retained. Reject takes priority over review; decision_reason_counts_by_status counts only codes for the winning severity.",
            "previous_turn": "Previous context is audit-only (exclude all prev_other_* columns from model inputs). It is the latest completed other-speaker turn, regardless of triage status.",
        },
    }


def safe_component(name):
    return isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name


def main():
    metadata = ROOT / "metadata"
    if metadata.is_symlink() or not metadata.is_dir():
        raise ValueError(f"Expected a real project metadata directory: {metadata}")
    manifest = pq.read_table(metadata / "manifest_sessions.parquet").to_pylist()
    manifest.sort(key=lambda row: row["conversation_id"])
    if len({m["conversation_id"] for m in manifest}) != len(manifest):
        raise ValueError("Duplicate conversation_id in manifest")
    rows = []
    no_transcript_count = 0
    for i, m in enumerate(manifest, 1):
        cid, a, b = m["conversation_id"], m["participant_a"], m["participant_b"]
        if not all(safe_component(p) for p in (cid, a, b)) or a == b:
            raise ValueError(f"Invalid manifest identifiers: {cid}, {a}, {b}")
        folder = ROOT / "clean_sessions" / cid
        try:
            pa_audio = load_participant(folder / f"{cid}_{a}", a)
            pb_audio = load_participant(folder / f"{cid}_{b}", b)
            duration = min(pa_audio.duration, pb_audio.duration)
            no_transcript_count += int(not pa_audio.transcript) + int(not pb_audio.transcript)
            turns = list(build_turns(m, "a", "b", pa_audio, pb_audio, duration))
            turns.extend(build_turns(m, "b", "a", pb_audio, pa_audio, duration))
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            raise RuntimeError(f"Failed to process {cid}: {error}") from error
        turns.sort(key=lambda r: (r["t_start"], r["speaker_role"], r["raw_transcript_start"]))
        for n, turn in enumerate(turns, len(rows) + 1):
            turn["turn_id"] = f"T_{n:06d}"
        attach_previous_turns(turns)
        rows.extend(turns)
        del pa_audio, pb_audio  # Bound the in-memory WAV data to a single pair.
        if i % 50 == 0 or i == len(manifest):
            print(f"Processed {i}/{len(manifest)} conversations; {len(rows)} turn candidates", flush=True)
    if not rows:
        raise ValueError("No transcript turns found in the manifest")
    report = summarize(rows, len(manifest), no_transcript_count)
    save_outputs(metadata, rows, report)
    print(f"Total candidates: {len(rows)}")
    for status in STATUSES:
        print(f"  {status}: {report['status_counts'][status]} ({report['status_percentages'][status]:.2f}%)"
              f" | buckets: {report['duration_bucket_by_status'][status]}")
    print(f"Wrote Step 2 CSV, Parquet and report under {metadata}")


if __name__ == "__main__":
    main()
