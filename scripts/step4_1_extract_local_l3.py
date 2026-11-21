#!/usr/bin/env python3
"""Extract local L3 audio and whole-face feature proxies for the calibration pilot."""

import argparse
import csv
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
SAMPLE_RATE = 48000
FAU_CHANNELS = 24
MIN_VALID_FRAMES = 10
MIN_VALID_RATIO = 0.5
MIN_Z_GROUP = 3
ONSET_SIGMAS = 1.5
CONTAM_ABS_MIN = 0.10
CONTAM_REL_MIN = 0.25
AUDIO_WIN_SEC = 0.04
AUDIO_HOP_SEC = 0.01
DBFS_FLOOR = -120.0
F0_RATE = 8000
F0_MIN, F0_MAX = 65.0, 500.0
F0_MIN_DBFS = -55.0
F0_MIN_CORRELATION = 0.65
F0_PEAK_FRACTION = 0.9
EPS = 1e-9
TURN_ID = re.compile(r"T_\d+")
SOURCE_ID = re.compile(r"[A-Za-z0-9_]+")
BASELINES = ("baseline_turn_prefix", "baseline_session_silent")
PRIMARY_BASELINE = "baseline_session_silent"
PROVENANCE = ("turn_id", "conversation_id", "vendor_id", "session_id", "prompt_hash", "prompt_id_unique",
              "speaker_id", "listener_id", "pilot_stratum", "triage_status", "duration_bucket")


def atomic_write(path, write, *, binary=False):
    opts = {"mode": "w+b"} if binary else {"mode": "w", "encoding": "utf-8", "newline": ""}
    temp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, **opts)
    staged = Path(temp.name)
    try:
        with temp:
            write(temp)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def json_write(file, data):
    json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write("\n")


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {"n": 0, **dict.fromkeys(("mean", "std", "min", "p10", "median", "p90", "max"))}
    p10, median, p90 = np.percentile(x, [10, 50, 90])
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "p10": float(p10), "median": float(median), "p90": float(p90), "max": float(x.max())}


def time_frames(samples, sample_rate):
    width, hop = round(AUDIO_WIN_SEC * sample_rate), round(AUDIO_HOP_SEC * sample_rate)
    if len(samples) < width:
        return np.empty((0, width), dtype=samples.dtype)
    return np.lib.stride_tricks.sliding_window_view(samples, width)[::hop]


def dbfs(power):
    return 10 * np.log10(np.maximum(power, 10 ** (DBFS_FLOOR / 10)))


def short_time_f0(waveform):
    """NSDF autocorrelation pitch estimate, using anti-aliased 8 kHz audio."""
    y = resample_poly(waveform, 1, SAMPLE_RATE // F0_RATE)
    frames = time_frames(y, F0_RATE)
    if not len(frames):
        return np.empty(0, dtype=np.float64)
    width = frames.shape[1]
    low, high = math.ceil(F0_RATE / F0_MAX), math.floor(F0_RATE / F0_MIN)
    lags = np.arange(high + 2)
    nfft = 1 << (2 * width - 2).bit_length()
    f0 = np.full(len(frames), np.nan, dtype=np.float64)
    for off in range(0, len(frames), 256):
        x = frames[off:off + 256].astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        power = x * x
        loud = dbfs(power.mean(axis=1)) >= F0_MIN_DBFS
        spectrum = np.fft.rfft(x, n=nfft, axis=1)
        ac = np.fft.irfft(np.abs(spectrum) ** 2, n=nfft, axis=1)[:, :len(lags)]
        csum = np.pad(power.cumsum(axis=1), ((0, 0), (1, 0)))
        norm = csum[:, width - lags] + csum[:, -1:] - csum[:, lags]
        corr = np.divide(2 * ac, norm, out=np.zeros_like(ac), where=norm > 0)
        for k in np.flatnonzero(loud):
            v = corr[k]
            maxima = np.flatnonzero((v[low:high + 1] > v[low - 1:high]) &
                                    (v[low:high + 1] >= v[low + 1:high + 2])) + low
            if not len(maxima):
                continue
            cutoff = max(F0_MIN_CORRELATION, F0_PEAK_FRACTION * v[maxima].max())
            qualified = maxima[v[maxima] >= cutoff]
            if not len(qualified):
                continue
            lag = qualified[0]
            left, top, right = v[lag - 1:lag + 2]
            denom = left - 2 * top + right
            shift = np.clip(0.5 * (left - right) / denom, -0.5, 0.5) if denom < 0 else 0.0
            hz = F0_RATE / (lag + shift)
            if F0_MIN <= hz <= F0_MAX:
                f0[off + k] = hz
    return f0


def extract_audio(wav_path, duration_sec, word_count, expected_samples):
    with wav_path.open("rb") as file, sf.SoundFile(file, mode="r") as wav:
        if wav.samplerate != SAMPLE_RATE or wav.channels != 1 or wav.subtype != "FLOAT":
            raise ValueError(f"Expected 48 kHz mono float32 speaker crop: {wav_path}")
        data = wav.read(dtype="float32")
    if len(data) != expected_samples or not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Invalid sample count or nonfinite samples in {wav_path}")
    x = data.astype(np.float64)
    frames = time_frames(x, SAMPLE_RATE)
    # Direct mean-square (no window weighting). Use a fixed floor for digital silence.
    frame_dbfs = dbfs(np.einsum("ij,ij->i", frames, frames) / frames.shape[1])
    levels = stats(frame_dbfs)
    pitch = short_time_f0(data)
    voiced = stats(pitch[np.isfinite(pitch)])
    return {
        "wav_sample_count": len(data), "wav_duration_sec": len(data) / SAMPLE_RATE,
        "rms_global_dbfs": float(dbfs(np.dot(x, x) / len(x))),
        "rms_frame_count": levels["n"], "rms_dbfs_mean": levels["mean"], "rms_dbfs_std": levels["std"],
        "rms_dbfs_p90": levels["p90"],
        "rms_dbfs_p90_p10": levels["p90"] - levels["p10"] if levels["n"] else None,
        "f0_frame_count": len(pitch), "f0_valid_frame_count": voiced["n"],
        "f0_valid_frame_ratio": voiced["n"] / len(pitch) if len(pitch) else 0.0,
        "f0_hz_mean": voiced["mean"], "f0_hz_std": voiced["std"],
        "f0_hz_p90_p10": voiced["p90"] - voiced["p10"] if voiced["n"] else None,
        "word_count": word_count, "words_per_sec": word_count / duration_sec,
    }


@dataclass
class Trace:
    fau: np.ndarray
    arousal: np.ndarray
    valence: np.ndarray
    valid: np.ndarray
    silent_baseline_mask: np.ndarray
    speaking: np.ndarray

    @property
    def n(self):
        return len(self.fau)


def frame_tick(t):
    return max(0, math.ceil(t * FPS - EPS))


def window_trace(trace, start, end):
    first, last = frame_tick(start), frame_tick(end)
    return slice(min(trace.n, first), min(trace.n, max(first, last))), max(0, last - first)


def source_trace(npz_file, json_file):
    with np.load(npz_file, allow_pickle=False) as z:
        fau = z["movement:FAUValue"]
        movement = z["movement:is_valid"].reshape(-1)
        arousal = z["movement:emotion_arousal"].reshape(-1)
        valence = z["movement:emotion_valence"].reshape(-1)
    if fau.shape != (len(movement), FAU_CHANNELS) or len(arousal) != len(movement) or len(valence) != len(movement):
        raise ValueError(f"Inconsistent NPZ feature dimensions in {npz_file}")
    with json_file.open(encoding="utf-8") as f:
        vad = json.load(f)["metadata:vad"]
    speaking = np.zeros(len(fau), dtype=bool)
    for v in vad:
        start, end = float(v["start"]), float(v["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError(f"Invalid VAD interval in {json_file}")
        speaking[frame_tick(start):frame_tick(end)] = True
    finite = np.isfinite(fau).all(axis=1) & np.isfinite(movement)
    return Trace(fau.astype(np.float64), arousal, valence, finite & (movement > 0.5),
                 finite & (movement == 1) & ~speaking, speaking)


def series_summary(x):
    if not len(x):
        return {"mean": None, "p90": None}
    return {"mean": float(x.mean()), "p90": float(np.percentile(x, 90))}


def emotion_summary(x, valid):
    x = x[valid & np.isfinite(x)]
    return {**series_summary(x), "range": float(np.ptp(x)) if len(x) else None}


def extract_speaker_visual(trace, start, end):
    sl, expected = window_trace(trace, start, end)
    valid = trace.valid[sl]
    n = int(valid.sum())
    ratio = n / expected if expected else 0.0
    observable = n >= MIN_VALID_FRAMES and ratio >= MIN_VALID_RATIO
    silent = trace.fau[trace.silent_baseline_mask]
    baseline_ok = len(silent) >= MIN_VALID_FRAMES
    fau = trace.fau[sl]
    # Require adjacent measured frames; never difference across a missing-frame gap.
    right_idxs = np.flatnonzero(valid[:-1] & valid[1:]) + 1 if observable else np.empty(0, dtype=int)
    changes = np.linalg.norm(fau[right_idxs] - fau[right_idxs - 1], axis=1)
    relative_peak = None
    if changes.size and changes.max() > 0:
        t_peak = (sl.start + right_idxs[int(np.argmax(changes))]) / FPS
        relative_peak = float(np.clip((t_peak - start) / (end - start), 0.0, 1.0))
    offset = (np.linalg.norm(fau[valid] - np.median(silent, axis=0), axis=1)
              if observable and baseline_ok else np.empty(0))
    stats_mask = valid if observable else np.zeros(len(valid), dtype=bool)
    return {
        "speaker_visual_observable": observable, "expected_frame_count": expected,
        "valid_frame_count": n, "valid_frame_ratio": ratio, "adjacent_valid_pair_count": len(changes),
        "session_silent_baseline_frame_count": len(silent), "session_silent_baseline_available": baseline_ok,
        "fau_frame_delta_l2": series_summary(changes), "fau_session_silent_offset_l2": series_summary(offset),
        "emotion_arousal": emotion_summary(trace.arousal[sl], stats_mask),
        "emotion_valence": emotion_summary(trace.valence[sl], stats_mask), "peak_rel_pos": relative_peak,
    }


def onset(times, distances, population, threshold, anchor):
    if threshold is None or not len(distances):
        return None
    ix = np.flatnonzero(population & (distances > threshold))
    return float(times[ix[0]] - anchor) if len(ix) else None


def baseline_comparison(silent, window, times, valid, non_speaking, window_ok, threshold, anchor):
    enough = len(silent) >= MIN_VALID_FRAMES
    observability = ("unobservable" if not window_ok or not enough else
                     "observable" if non_speaking.any() else "speaking_contaminated")
    median = np.median(silent, axis=0) if enough else None
    l2 = l1 = np.empty(0)
    use_valid = np.empty(0, dtype=bool)
    use_times = np.empty(0)
    if observability != "unobservable":
        delta = window[valid] - median
        l2, l1 = np.linalg.norm(delta, axis=1), np.abs(delta).sum(axis=1)
        use_valid, use_times = non_speaking[valid], times[valid]
    a, q = series_summary(l2), series_summary(l2[use_valid])
    a1, q1 = series_summary(l1), series_summary(l1[use_valid])
    return {
        "valid_frame_count": len(silent), "baseline_available": enough,
        "median_fau_24": median.tolist() if median is not None else None,
        "listener_observability": observability,
        "delta_all_mean": a["mean"], "delta_all_p90": a["p90"],
        "delta_non_speaking_mean": q["mean"], "delta_non_speaking_p90": q["p90"],
        "delta_all_l1_mean": a1["mean"], "delta_all_l1_p90": a1["p90"],
        "delta_non_speaking_l1_mean": q1["mean"], "delta_non_speaking_l1_p90": q1["p90"],
        "reaction_onset_rel_end_sec": onset(use_times, l2, use_valid, threshold, anchor),
        "reaction_onset_all_rel_end_sec": onset(use_times, l2, np.ones(len(l2), dtype=bool), threshold, anchor),
    }


def extract_listener(trace, start, end, ref_start, ref_end):
    reaction_slice, expected = window_trace(trace, ref_start, ref_end)
    valid = trace.valid[reaction_slice]
    speaking = trace.speaking[reaction_slice]
    quiet = valid & ~speaking
    times = np.arange(reaction_slice.start, reaction_slice.stop) / FPS
    count = int(valid.sum())
    ratio = count / expected if expected else 0.0
    window_ok = count >= MIN_VALID_FRAMES and ratio >= MIN_VALID_RATIO
    prefix_end = end - 1.0
    pre_slice, _ = window_trace(trace, start, prefix_end)
    prefix = trace.fau[pre_slice][trace.silent_baseline_mask[pre_slice]]
    session = trace.fau[trace.silent_baseline_mask]
    session_mean = session_std = threshold = None
    if len(session) >= MIN_VALID_FRAMES:
        radii = np.linalg.norm(session - np.median(session, axis=0), axis=1)
        session_mean, session_std = float(radii.mean()), float(radii.std())
        threshold = session_mean + ONSET_SIGMAS * session_std
    prefix_res = baseline_comparison(prefix, trace.fau[reaction_slice], times, valid, quiet,
                                     window_ok, threshold, end)
    session_res = baseline_comparison(session, trace.fau[reaction_slice], times, valid, quiet,
                                      window_ok, threshold, end)
    return {
        "listener_observability": session_res["listener_observability"], "primary_baseline": PRIMARY_BASELINE,
        "reaction_window_start_sec": ref_start, "reaction_window_end_sec": ref_end,
        "reaction_expected_frame_count": expected, "reaction_valid_frame_count": count,
        "reaction_valid_frame_ratio": ratio, "reaction_non_speaking_valid_frame_count": int(quiet.sum()),
        "reaction_speaking_valid_frame_count": int((speaking & valid).sum()),
        "reaction_vad_ratio_of_present_frames": float(speaking.mean()) if len(speaking) else None,
        "prefix_requested_start_sec": start, "prefix_requested_end_sec": prefix_end,
        "session_silent_delta_l2_mean": session_mean, "session_silent_delta_l2_std": session_std,
        "onset_l2_threshold": threshold,
        "baseline_turn_prefix": prefix_res, "baseline_session_silent": session_res,
    }


# Each normalization metric uses the participant appropriate to its role.
SPEAKER = "speaker_stimulus"
LISTENER = "listener_reaction"
NORMALIZE = {
    "speaker_rms_frame_std_db": ("speaker", (SPEAKER, "audio", "rms_dbfs_std")),
    "speaker_rms_frame_p90_p10_db": ("speaker", (SPEAKER, "audio", "rms_dbfs_p90_p10")),
    "speaker_f0_span_hz": ("speaker", (SPEAKER, "audio", "f0_hz_p90_p10")),
    **{f"speaker_fau_{feature}_{agg}": ("speaker", (SPEAKER, "visual", feature, agg))
       for feature in ("fau_frame_delta_l2", "fau_session_silent_offset_l2") for agg in ("mean", "p90")},
    **{f"listener_{baseline}_{population}_{agg}": ("listener", (LISTENER, baseline, f"delta_{population}_{agg}"))
       for baseline in BASELINES for population in ("all", "non_speaking") for agg in ("mean", "p90")},
}


def get_path(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def add_normalizations(records):
    diagnostics = {}
    for r in records:
        r["feature"]["normalized"] = {"participant_z": {}, "participant_session_z": {}}
    for name, (role, path) in NORMALIZE.items():
        available = []
        for record in records:
            value = get_path(record["feature"], path)
            for zkey in ("participant_z", "participant_session_z"):
                record["feature"]["normalized"][zkey][name] = None
            if value is not None:
                available.append((record, value))
        detail = {"role": role, "raw_global": stats([v for _, v in available])}
        for zkey, with_session in (("participant_z", False), ("participant_session_z", True)):
            groups = defaultdict(list)
            for record, value in available:
                row = record["row"]
                group = (row["vendor_id"], row[f"{role}_id"])
                if with_session:
                    group += (row["session_id"],)
                groups[group].append((record, value))
            group_stats, normalized = [], []
            for key, rows in sorted(groups.items()):
                values = np.array([v for _, v in rows], dtype=np.float64)
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else None
                group_stats.append({"vendor_id": key[0], "participant_id": key[1],
                                    **({"session_id": key[2]} if with_session else {}),
                                    "n": len(rows), "raw_mean": mean, "raw_std_sample": sd})
                if len(values) >= MIN_Z_GROUP and sd is not None and sd > 1e-12:
                    for (record, _), z in zip(rows, (values - mean) / sd):
                        record["feature"]["normalized"][zkey][name] = float(z)
                        normalized.append(float(z))
            detail[zkey] = {"distribution": stats(normalized), "group_count": len(group_stats),
                            "groups_normalized": sum(g["n"] >= MIN_Z_GROUP and (g["raw_std_sample"] or 0) > 1e-12
                                                     for g in group_stats)}
            if not with_session:
                eligible = [g for g in group_stats if g["n"] >= MIN_Z_GROUP]
                positive = [g for g in eligible if g["raw_mean"] > 0]
                lo = min(positive, key=lambda g: g["raw_mean"], default=None)
                hi = max(positive, key=lambda g: g["raw_mean"], default=None)
                detail["participant_group_means"] = group_stats
                detail["participant_raw_mean_fold"] = {
                    "eligible_positive_groups": len(positive), "excluded_zero_groups": len(eligible) - len(positive),
                    "min_group": lo, "max_group": hi,
                    "max_over_min_positive_mean": hi["raw_mean"] / lo["raw_mean"] if len(positive) >= 2 else None,
                }
        diagnostics[name] = detail
    return diagnostics


def paired_spearman(records, mean_path, p90_path):
    points = [(r["feature"]["duration_sec"], get_path(r["feature"], mean_path), get_path(r["feature"], p90_path))
              for r in records]
    points = np.asarray([(d, m, p) for d, m, p in points if m is not None and p is not None], dtype=float)
    result = {"n": len(points), "mean_rho": None, "p90_rho": None, "smaller_abs_rho": None}
    if len(points) >= 3:
        for idx, agg in enumerate(("mean", "p90"), 1):
            if np.ptp(points[:, 0]) > 0 and np.ptp(points[:, idx]) > 0:
                result[f"{agg}_rho"] = float(spearmanr(points[:, 0], points[:, idx]).statistic)
        m, p = result["mean_rho"], result["p90_rho"]
        if m is not None and p is not None:
            result["smaller_abs_rho"] = "mean" if abs(m) < abs(p) else "p90" if abs(p) < abs(m) else "tie"
    return result


def duration_diagnostics(records):
    pairs = {name: ((SPEAKER, "visual", name, "mean"), (SPEAKER, "visual", name, "p90"))
             for name in ("fau_frame_delta_l2", "fau_session_silent_offset_l2")}
    pairs["speaker_frame_rms_dbfs"] = ((SPEAKER, "audio", "rms_dbfs_mean"), (SPEAKER, "audio", "rms_dbfs_p90"))
    for baseline in BASELINES:
        for population in ("all", "non_speaking"):
            pairs[f"listener_{baseline}_{population}"] = ((LISTENER, baseline, f"delta_{population}_mean"),
                                                           (LISTENER, baseline, f"delta_{population}_p90"))
    return {name: paired_spearman(records, *paths) for name, paths in pairs.items()}


def baseline_availability(listeners):
    out = {"total": len(listeners)}
    for baseline in BASELINES:
        rows = [listener[baseline] for listener in listeners]
        n = sum(row["baseline_available"] for row in rows)
        out[baseline] = {"available": n, "missing": len(rows) - n,
                         "available_rate": n / len(rows) if rows else None,
                         "missing_rate": (len(rows) - n) / len(rows) if rows else None,
                         "observability_counts": dict(Counter(r["listener_observability"] for r in rows))}
    return out


def contamination(listeners, baseline):
    paired, voiced, signed, paired_voice_rel = [], 0, [], []
    above = below = 0
    for listener in listeners:
        row = listener[baseline]
        all_delta, silent_delta = row["delta_all_p90"], row["delta_non_speaking_p90"]
        if all_delta is None or silent_delta is None:
            continue
        diff = all_delta - silent_delta
        paired.append(diff)
        if not listener["reaction_speaking_valid_frame_count"]:
            continue
        voiced += 1
        signed.append(diff)
        rel = abs(diff) / max(silent_delta, 1e-12)
        paired_voice_rel.append(rel)
        if abs(diff) >= CONTAM_ABS_MIN and rel >= CONTAM_REL_MIN:
            if diff > 0:
                above += 1
            else:
                below += 1
    changed = above + below
    return {
        "paired_count": len(paired), "paired_with_b_speaking_count": voiced,
        "not_comparable_count": len(listeners) - len(paired),
        "speaking_contaminated_no_silent_valid_count":
            sum(x[baseline]["listener_observability"] == "speaking_contaminated" for x in listeners),
        "material_change_count": changed, "higher_with_speech_count": above, "lower_with_speech_count": below,
        "material_change_rate_of_paired": changed / len(paired) if paired else None,
        "material_change_rate_of_paired_with_speech": changed / voiced if voiced else None,
        "material_change_rate_of_all": changed / len(listeners) if listeners else None,
        "signed_difference_stats_all_paired": stats(paired),
        "signed_difference_stats_paired_with_speech": stats(signed),
        "relative_absolute_difference_stats_paired_with_speech": stats(paired_voice_rel),
    }


def safe_file(path, expected):
    if path.is_symlink() or path.resolve(strict=True) != expected or not path.is_file():
        raise ValueError(f"Unexpected file path: {path} (expected {expected})")
    return path


def sample_record(row):
    tid, cid = row["turn_id"], row["conversation_id"]
    if not TURN_ID.fullmatch(tid) or not SOURCE_ID.fullmatch(cid):
        raise ValueError(f"Invalid pilot IDs: {tid}, {cid}")
    turn_root = ROOT / "cropped_turns" / tid
    if turn_root.resolve(strict=True) != turn_root:
        raise ValueError(f"Noncanonical crop directory: {turn_root}")
    mp = safe_file(ROOT / row["meta_json_path"], turn_root / "meta.json")
    wav = safe_file(ROOT / row["speaker_wav_path"], turn_root / "speaker_A.wav")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    for key in ("turn_id", "conversation_id", "speaker_id", "listener_id"):
        if row[key] != meta["source_provenance"][key]:
            raise ValueError(f"Index and meta disagree on {key}")
    profile = meta.get("l3_profile")
    if profile is not None and not isinstance(profile, dict):
        raise ValueError("Malformed L3 profile")
    sources = meta["audit_refs"]["source_files"]
    traces = {}
    for role in ("speaker", "listener"):
        pid = row[f"{role}_id"]
        if not SOURCE_ID.fullmatch(pid):
            raise ValueError(f"Invalid participant: {pid}")
        stem = ROOT / "clean_sessions" / cid / f"{cid}_{pid}"
        npz_file = safe_file(turn_root / sources[role]["npz"], stem.with_suffix(".npz"))
        json_file = safe_file(turn_root / sources[role]["json"], stem.with_suffix(".json"))
        traces[role] = source_trace(npz_file, json_file)
    m = meta["crop_mapping"]
    start, end, ref_start, ref_end, full = (float(m[key]) for key in
                                         ("raw_t_start", "raw_t_end", "ref_b_start", "ref_b_end", "time_axis_end_sec"))
    if (not (0 <= start < end <= full and 0 <= ref_start < ref_end <= full)
            or not math.isclose(ref_start, max(0, end - 0.5), abs_tol=EPS)
            or not math.isclose(ref_end, min(full, end + 2), abs_tol=EPS)):
        raise ValueError("Invalid windows in crop_mapping")
    duration = float(meta["triage"]["duration_sec"])
    if not math.isclose(duration, end - start, abs_tol=EPS):
        raise ValueError("Duration disagrees with crop_mapping")
    audio = extract_audio(wav, duration, int(meta["triage"]["word_count"]), int(m["speaker_wav"]["sample_count"]))
    feature = {
        "schema_version": 1,
        "duration_sec": duration,
        "fau_channel_names_verified": False,
        SPEAKER: {"audio": audio, "visual": extract_speaker_visual(traces["speaker"], start, end)},
        LISTENER: extract_listener(traces["listener"], start, end, ref_start, ref_end),
    }
    return {"row": row, "path": mp, "meta": meta, "feature": feature}


def flatten(obj, prefix=""):
    record = {}
    for key, value in obj.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            record.update(flatten(value, name))
        else:
            record[name] = value
    return record


def write_results(metadata, records, report, stem):
    rows = [{**{k: r["row"][k] for k in PROVENANCE}, **flatten(r["feature"])} for r in records]
    table = pa.Table.from_pylist(rows)
    atomic_write(metadata / f"{stem}_l3_features.parquet", lambda f: pq.write_table(table, f), binary=True)

    def write_csv(f):
        out = csv.DictWriter(f, fieldnames=table.column_names)
        out.writeheader()
        out.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)

    atomic_write(metadata / f"{stem}_l3_features.csv", write_csv)
    for record in records:
        meta = json.loads(record['path'].read_text())
        profile = dict(meta['l3_profile'] or {})
        profile.setdefault('multimodal_review', None)
        profile['feature_based'] = record['feature']
        meta['l3_profile'] = profile
        atomic_write(record["path"], lambda f: json_write(f, meta))
    report_name = 'step4_1_stability_report.json' if stem=='pilot_180' else f'step4_1_{stem}_report.json'
    atomic_write(metadata / report_name, lambda f: json_write(f, report))


def main():
    metadata = ROOT / "metadata"
    if metadata.is_symlink() or not metadata.is_dir():
        raise ValueError(f"Unsafe metadata directory: {metadata}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-index', type=Path, default=metadata/'pilot_180_index.parquet')
    src = parser.parse_args().input_index.resolve(strict=True)
    if not src.name.endswith('_index.parquet'):
        raise ValueError('Expected *_index.parquet')
    stem = src.name.removesuffix('_index.parquet')
    index = pq.read_table(src).to_pylist()
    if (not index or len({r['turn_id'] for r in index}) != len(index)
            or any(r['crop_status'] not in ('success','reused') for r in index)):
        raise ValueError('Expected unique, complete crops')
    if stem == 'pass_1561' and (len(index)!=1561 or any(r['triage_status']!='pass' for r in index)):
        raise ValueError('Incomplete production index')
    records = []
    for i, row in enumerate(sorted(index, key=lambda r: r["turn_id"]), 1):
        try:
            records.append(sample_record(row))
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            raise RuntimeError(f"L3 extraction failed for {row['turn_id']}: {error}") from error
        if i % 30 == 0:
            print(f"Extracted {i}/{len(index)} feature profiles", flush=True)
    listeners = [r["feature"][LISTENER] for r in records]
    short = [r["feature"][LISTENER] for r in records if r["row"]["duration_bucket"] == "short"]
    speaker_visual_n = sum(r["feature"][SPEAKER]["visual"]["speaker_visual_observable"] for r in records)
    report = {
        "feature_mode": "feature_based",
        "source_index": str(src), "samples": len(records),
        "fau_channel_names_verified": False, "fau_dimensions_used": FAU_CHANNELS,
        "fau_channel_note": "No verified FAU column map in the local artifacts. Whole-face distances only; no upper/lower-face split.",
        "speaker_visual_observable": {"true": speaker_visual_n, "false": len(records) - speaker_visual_n},
        "listener_primary_observability_counts": dict(Counter(r["listener_observability"] for r in listeners)),
        "listener_baseline_availability": {"all": baseline_availability(listeners), "short_lt_2_sec": baseline_availability(short)},
        "spearman_duration_mean_vs_p90": duration_diagnostics(records),
        "vocal_contamination_all_vs_non_speaking_l2_p90": {key: contamination(listeners, key) for key in BASELINES},
        "raw_vs_normalized_and_participant_means": add_normalizations(records),
        "method": {
            "audio": "Global mono RMS plus 40 ms RMS frames every 10 ms; dBFS floor -120. F0 is an autocorrelation proxy (8 kHz anti-aliased, 65–500 Hz, 40/10 ms, minimum -55 dBFS and normalized correlation 0.65).",
            "npz_time": "NPZ frame i has nominal master time i/30 s; windows are half-open. Denominators include expected frames beyond available data. Valid requires movement>0.5 and finite 24-D FAU; silent baseline strictly uses movement==1 and own non-VAD frames.",
            "observability": "Require >=10 valid reaction/visual frames and >=50% valid, and >=10 baseline frames for listener metrics. With zero quiet valid reaction frames, keep the all-frame diagnostic only and mark speaking_contaminated.",
            "speaker_peak_rel_pos": "Location of maximum whole-face adjacent-valid-frame FAU L2 change (later frame time), not a semantic or audiovisual stimulus annotation.",
            "listener_distance": "L2 fields have unqualified delta names; L1 uses _l1_. Onset is the first frame whose L2 distance to the chosen baseline median exceeds the session-silent mean distance to its own median plus 1.5 population SD. Primary onset excludes B-VAD frames.",
            "material_change": "An operational (not significance-test) threshold: B spoke in at least one valid frame and |P90(all)-P90(quiet)| >= 0.10 FAU L2 units and >=25% of P90(quiet). Counts report both conditional and whole-pilot denominators.",
            "z_score": "Within role and (vendor, participant), and additionally within role and (vendor, participant, session); require >=3 nonnull rows and nonzero sample SD. Means and scales are fit on all samples of the input index, not for train/test reuse.",
            "participant_fold": "Maximum/minimum positive raw participant means among groups with >=3 observations; signed z scores have no fold ratio.",
            "spearman": "Mean and P90 each correlated with turn duration on the same complete-case subset. Smaller absolute rho denotes weaker observed monotonic association, not a significance test.",
        },
    }
    write_results(metadata, records, report, stem)
    print(f"Wrote {len(records)} L3 profiles; speaker visual {speaker_visual_n} observable, "
          f"listener (session baseline) {report['listener_primary_observability_counts']}")


if __name__ == "__main__":
    main()
