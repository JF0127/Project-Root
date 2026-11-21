#!/usr/bin/env python3
"""Select a stratified calibration pilot and render WAV-clock-aligned media crops."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261120
SAMPLE_RATE = 48_000
EPS = Fraction(1, 10**9)
PTS_ZERO_TOLERANCE = 1e-6
CROP_VERSION = 2
PRESET = "veryfast"
CRF = 21
QUOTAS = {
    "pass_core": 70, "pass_short": 20, "pass_long": 20,
    "review_late_overlap": 25, "review_moderate_overlap": 15,
    "review_move_invalid": 20, "review_low_crosstalk_snr": 10,
}
FALLBACK_LABEL = "pass_core_fallback"
COVERAGE_KEYS = ("session_id", "speaker_id", "prompt_id_unique")
PROVENANCE_KEYS = ("turn_id", "conversation_id", "vendor_id", "session_id", "prompt_hash",
                   "prompt_id_unique", "speaker_id", "listener_id")
OUTPUTS = ("speaker_A.wav", "speaker_A.mp4", "listener_B_ref.mp4", "meta.json")
META_BLOCKS = {"model_inputs", "audit_refs", "source_provenance", "crop_mapping", "triage", "l3_profile"}
TURN_ID = re.compile(r"T_\d+")
SOURCE_ID = re.compile(r"[A-Za-z0-9_]+")
SHOWINFO_TB = re.compile(r"config in time_base:\s*(\d+/\d+)")
SHOWINFO_FRAME = re.compile(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:")
DEFAULT_WORKERS = min(6, max(1, (os.cpu_count() or 1) // 3))
TRIAGE_METRICS = (
    "duration_sec", "duration_bucket", "word_count", "has_terminal_punct", "is_minimal_filler",
    "transcript_vad_overlap_ratio", "backchannel_overlap_sec", "substantive_overlap_sec",
    "substantive_overlap_ratio", "late_overlap_flag", "speaker_rms_db", "listener_silent_rms_db",
    "crosstalk_snr_db", "listener_ref_vad_ratio",
    "speaker_box_valid_ratio", "speaker_box_max_consec_invalid", "speaker_move_valid_ratio",
    "speaker_move_max_consec_invalid", "listener_box_valid_ratio", "listener_box_max_consec_invalid",
    "listener_move_valid_ratio", "listener_move_max_consec_invalid",
)


def output_directory(path):
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Unsafe project output directory: {path}")
    if path.resolve().is_relative_to((ROOT / "clean_sessions").resolve()):
        raise ValueError(f"Refusing to write inside clean_sessions: {path}")
    path.mkdir(exist_ok=True)
    return path


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


def write_json(file, data):
    json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write("\n")


def write_table(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty table")
    table = pa.Table.from_pylist(rows)
    atomic_write(path.with_suffix(".parquet"), lambda file: pq.write_table(table, file), binary=True)

    def write_csv(file):
        out = csv.DictWriter(file, fieldnames=table.column_names)
        out.writeheader()
        out.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                       for k, v in row.items()} for row in rows)

    atomic_write(path.with_suffix(".csv"), write_csv)


def coverage(rows):
    return {field: len({row[field] for row in rows if row.get(field)})
            for field in (*COVERAGE_KEYS, "conversation_id")}


def validate_turn_ids(rows):
    ids = [row.get("turn_id") for row in rows]
    if (not ids or len(ids) != len(set(ids))
            or any(not isinstance(tid, str) or not TURN_ID.fullmatch(tid) for tid in ids)):
        raise ValueError("Expected a nonempty table with unique T_<digits> turn IDs")


def strata_for(row):
    if row["triage_status"] == "pass":
        yield "pass_" + row["duration_bucket"]
    elif row["triage_status"] == "review":
        codes = set(row["reason_codes"])
        has_move_issue = any(code.startswith(("SPEAKER_MOVE_", "LISTENER_MOVE_")) for code in codes)
        moderate = "MODERATE_SUBSTANTIVE_OVERLAP" in codes
        if "LATE_SUBSTANTIVE_OVERLAP" in codes and not has_move_issue and not moderate:
            yield "review_late_overlap"
        if moderate:
            yield "review_moderate_overlap"
        if has_move_issue:
            yield "review_move_invalid"
        if "LOW_CROSSTALK_SNR" in codes:
            yield "review_low_crosstalk_snr"


def stratified_sample(rows, quotas=QUOTAS, seed=SEED):
    """Fill the least flexible strata first, preferring core review and novel IDs."""
    validate_turn_ids(rows)
    rows_by_id = {row["turn_id"]: row for row in rows}
    pools = {name: [] for name in quotas}
    rng = random.Random(seed)
    ties = {}
    for tid, row in sorted(rows_by_id.items()):
        ties[tid] = rng.random()
        for name in strata_for(row):
            if name in pools:
                pools[name].append(tid)
    assigned, fallback_for = {}, {}
    used = {key: Counter() for key in (*COVERAGE_KEYS, "conversation_id")}
    counts = Counter()

    def score(tid, name):
        row = rows_by_id[tid]
        repeats = [used[key][row[key]] for key in COVERAGE_KEYS]
        return (not name.startswith("review_") or row["duration_bucket"] == "core",
                sum(n == 0 for n in repeats), -used["conversation_id"][row["conversation_id"]],
                -sum(repeats), ties[tid])

    def add(tid, name):
        if tid not in assigned:
            for key in used:
                used[key][rows_by_id[tid][key]] += 1
        else:
            counts[assigned[tid]] -= 1
        assigned[tid] = name
        counts[name] += 1

    while True:
        available = {name: [tid for tid in pool if tid not in assigned] for name, pool in pools.items()
                     if counts[name] < quotas[name]}
        viable = [name for name, ids in available.items() if ids]
        if not viable:
            break
        name = min(viable, key=lambda key: len(available[key]) / (quotas[key] - counts[key]))
        add(max(available[name], key=lambda tid: score(tid, name)), name)

    def augment(name, seen_groups):
        # If two review strata share candidates, reclaim an assigned row only if
        # its previous stratum can be replenished without reducing any quota.
        seen_groups.add(name)
        for tid in sorted(pools[name], key=lambda tid: score(tid, name), reverse=True):
            old = assigned.get(tid)
            if old is None or (old not in seen_groups and augment(old, seen_groups)):
                add(tid, name)
                return True
        return False

    for name in quotas:
        while counts[name] < quotas[name] and augment(name, set()):
            pass
    shortages = {name: quotas[name] - counts[name] for name in quotas if counts[name] < quotas[name]}
    if shortages:
        spare = {tid for tid, row in rows_by_id.items()
                 if row["triage_status"] == "pass" and row["duration_bucket"] == "core" and tid not in assigned}
        if len(spare) < sum(shortages.values()):
            raise ValueError("Not enough unselected pass/core candidates to fill the pilot quotas")
        for name, missing in shortages.items():
            for _ in range(missing):
                tid = max(spare, key=lambda tid: score(tid, FALLBACK_LABEL))
                spare.remove(tid)
                add(tid, FALLBACK_LABEL)
                fallback_for[tid] = name
    result = [{**rows_by_id[tid], "pilot_stratum": name, "pilot_fallback_for": fallback_for.get(tid)}
              for tid, name in sorted(assigned.items())]
    return result, shortages, {name: len(pools[name]) for name in quotas}


def select_pilot(metadata):
    rows = pq.read_table(metadata / "turns_candidates.parquet").to_pylist()
    picked, shortages, available = stratified_sample(rows)
    selection = metadata / "pilot_180_selection"
    report = {
        "seed": SEED, "selection_count": len(picked), "target_quotas": QUOTAS,
        "eligible_count_by_stratum": available,
        "actual_strata": dict(Counter(row["pilot_stratum"] for row in picked)),
        "review_duration_buckets": dict(Counter(row["duration_bucket"] for row in picked
                                                if row["triage_status"] == "review")),
        "fallback_from_pass_core": shortages, "coverage": coverage(picked),
        "eligible_pool_coverage": coverage([row for row in rows if row["triage_status"] in ("pass", "review")]),
        "late_overlap_stratum": "Has late overlap, but neither moderate overlap nor movement-mask warnings. Other review warnings are allowed.",
    }
    write_table(selection, picked)
    atomic_write(metadata / "step3_pilot_180_selection_report.json", lambda file: write_json(file, report))
    print(f"Selected {len(picked)} turns: {report['actual_strata']}")
    print(f"Coverage: {report['coverage']}; fallback from pass/core: {shortages or 'none'}")


@dataclass(frozen=True)
class VideoProbe:
    time_base: Fraction
    start_pts: int
    duration: float
    frame_count: int
    avg_fps: Fraction

    @property
    def start_sec(self):
        return float(self.start_pts * self.time_base)


def probe_video(ffprobe, path, *, preview=False):
    args = [ffprobe, "-v", "error", "-show_entries",
            "stream=codec_type,codec_name,pix_fmt,channels,sample_rate,time_base,start_pts,duration_ts,"
            "duration,nb_frames,avg_frame_rate:format=duration", "-of", "json", str(path)]
    if not preview:
        args[3:3] = ["-select_streams", "v:0"]
    process = subprocess.run(args, capture_output=True, text=True, check=True, timeout=30)
    if process.stderr.strip():
        raise ValueError(f"ffprobe reported problems in {path.name}: {process.stderr.strip()}")
    data = json.loads(process.stdout)
    videos = [s for s in data["streams"] if s["codec_type"] == "video"]
    if len(videos) != 1:
        raise ValueError(f"Expected exactly one video stream in {path.name}")
    stream = videos[0]
    time_base = Fraction(stream["time_base"])
    video = VideoProbe(time_base=time_base, start_pts=int(stream["start_pts"]),
                       duration=stream_seconds(stream), frame_count=int(stream["nb_frames"]),
                       avg_fps=Fraction(stream["avg_frame_rate"]))
    if video.duration <= 0 or video.frame_count <= 0 or video.avg_fps <= 0:
        raise ValueError(f"Invalid video timing metadata in {path.name}")
    if not preview:
        return video
    audios = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if (len(audios) != 1 or audios[0]["codec_name"] != "aac" or audios[0]["channels"] != 1
            or int(audios[0]["sample_rate"]) != SAMPLE_RATE or stream["codec_name"] != "h264"
            or stream["pix_fmt"] != "yuv420p"):
        raise ValueError(f"Unexpected preview codecs, sample rate or channels in {path.name}")
    return video, {
        "container_duration_sec": float(data["format"]["duration"]),
        "preview_audio_duration_sec": stream_seconds(audios[0]),
        "preview_audio_start_pts": int(audios[0]["start_pts"]),
        "preview_audio_start_sec": float(int(audios[0]["start_pts"]) * Fraction(audios[0]["time_base"])),
    }


def stream_seconds(stream):
    if stream.get("duration_ts") not in (None, "N/A"):
        return float(int(stream["duration_ts"]) * Fraction(stream["time_base"]))
    return float(stream["duration"])


def cut_wav(source, dest, start, end):
    with source.open("rb") as file, sf.SoundFile(file, mode="r") as wav:
        rate = wav.samplerate
        if rate != SAMPLE_RATE or wav.channels != 1 or wav.subtype != "FLOAT":
            raise ValueError(f"Expected a mono {SAMPLE_RATE} Hz IEEE float WAV: {source}")
        first, last = round(start * rate), round(end * rate)
        if not 0 <= first < last <= wav.frames:
            raise ValueError(f"WAV crop out of bounds: [{first}, {last}) of {wav.frames} in {source}")
        wav.seek(first)
        samples = wav.read(last - first, dtype="float32")
    if len(samples) != last - first:
        raise ValueError(f"Short WAV read: {source}")
    sf.write(dest, samples, rate, format="WAV", subtype="FLOAT")
    info = sf.info(dest)
    if (info.frames, info.samplerate, info.channels, info.subtype) != (last - first, rate, 1, "FLOAT"):
        raise ValueError(f"Written WAV does not match sample-exact crop: {dest}")
    return {"start_sample": first, "end_sample_exclusive": last, "sample_count": info.frames,
            "sample_rate_hz": rate, "channels": info.channels, "wav_format": "pcm_f32le",
            "duration_sec": info.frames / rate, "local_zero_to_master_sec": first / rate}


def cut_video(ffmpeg, ffprobe, source, preview_wav, dest, start, end, wav_mapping, source_video):
    # A pair of ceil() bounds matches half-open master-clock intervals in the
    # original source ticks. A 1 ns tolerance only neutralizes float timestamp noise.
    tb = source_video.time_base
    start_tick = math.ceil((Fraction(str(start)) - EPS) / tb)
    end_tick = math.ceil((Fraction(str(end)) - EPS) / tb)
    vf = f"trim=start_pts={start_tick}:end_pts={end_tick},showinfo=checksum=0,setpts=PTS-STARTPTS"
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-nostats", "-y", "-copyts",
               "-filter_threads", "1"]
    # Seek before the crop to avoid decoding the entire conversation. Keep all
    # decoder preroll and use absolute seek timestamps, not start-PTS-relative seeks.
    if start > 2:
        command.extend(["-seek_timestamp", "1", "-ss", f"{start - 2:.9f}", "-noaccurate_seek"])
    command.extend(["-threads:v", "2", "-i", str(source), "-i", str(preview_wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-vf", vf, "-vsync", "1",
                    "-r:v", str(source_video.avg_fps),
                    "-c:v", "libx264", "-threads:v", "2", "-preset", PRESET, "-crf", str(CRF),
                    "-pix_fmt", "yuv420p",
                    "-video_track_timescale", str(tb.denominator), "-c:a", "aac", "-b:a", "128k",
                    "-use_editlist", "1", "-avoid_negative_ts", "disabled", "-movflags", "+faststart", str(dest)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise ValueError(f"FFmpeg exited {result.returncode} for {dest.name}: {result.stderr[-3000:]}")
    frame_pts = [int(tick) for tick in SHOWINFO_FRAME.findall(result.stderr)]
    filter_bases = {Fraction(base) for base in SHOWINFO_TB.findall(result.stderr)}
    if (not frame_pts or filter_bases != {tb} or min(frame_pts) < start_tick or max(frame_pts) >= end_tick
            or any(a >= b for a, b in zip(frame_pts, frame_pts[1:]))):
        raise ValueError(f"Could not verify actual source PTS for {dest.name}")
    # This corpus is CFR. Verify the selected source PTS grid explicitly so the
    # encode cannot silently retime irregular source frames. FFmpeg 4.4's CFR
    # muxing also supplies a correct duration for the final video frame.
    frame_period = 1 / source_video.avg_fps
    if any(abs((tick - frame_pts[0]) * tb - n * frame_period) > tb for n, tick in enumerate(frame_pts)):
        raise ValueError(f"Selected frames are not on the source CFR grid: {source.name}")
    video, muxed = probe_video(ffprobe, dest, preview=True)
    if video.frame_count != len(frame_pts) or abs(video.start_sec) > PTS_ZERO_TOLERANCE:
        raise ValueError(f"Cropped video frame count or local start does not match the filtered frames: {dest}")
    if abs(muxed["preview_audio_start_sec"]) > PTS_ZERO_TOLERANCE:
        raise ValueError(f"Preview audio did not start at local zero: {dest}")
    first_frame_master = float(frame_pts[0] * tb)
    video_audio_delta = abs(video.duration - wav_mapping["duration_sec"])
    if video_audio_delta > float(frame_period) + 1 / SAMPLE_RATE + PTS_ZERO_TOLERANCE:
        raise ValueError(f"Cropped video duration differs from the audio crop by more than one frame: {dest}")
    if abs(muxed["preview_audio_duration_sec"] - wav_mapping["duration_sec"]) > 0.002:
        raise ValueError(f"Muxed AAC duration differs from the WAV slice by more than 2 ms: {dest}")
    return {
        "requested_start_sec": start, "requested_end_sec": end,
        "source_video_start_pts": source_video.start_pts, "source_video_start_sec": source_video.start_sec,
        "source_video_time_base": str(tb), "source_first_kept_pts": frame_pts[0],
        "source_last_kept_pts": frame_pts[-1], "source_first_kept_sec": first_frame_master,
        "source_last_kept_sec": float(frame_pts[-1] * tb),
        "video_local_zero_to_master_sec": first_frame_master,
        "preview_audio_local_zero_to_master_sec": wav_mapping["local_zero_to_master_sec"],
        "video_minus_audio_local_zero_master_sec": first_frame_master - wav_mapping["local_zero_to_master_sec"],
        "video_duration_sec": video.duration, "video_frame_count": video.frame_count,
        "video_start_pts": video.start_pts, "video_start_sec": video.start_sec,
        "video_time_base": str(video.time_base), "video_wav_abs_duration_diff_sec": video_audio_delta,
        "container_wav_abs_duration_diff_sec": abs(muxed["container_duration_sec"] - wav_mapping["duration_sec"]),
        **muxed,
    }


def master_windows(row):
    start, end, ref_start, ref_end, limit = (float(row[key]) for key in
                                          ("t_start", "t_end", "ref_b_start", "ref_b_end", "time_axis_end_sec"))
    if (not all(math.isfinite(n) for n in (start, end, ref_start, ref_end, limit))
            or not (0 <= start < end <= limit and 0 <= ref_start < ref_end <= limit)
            or not math.isclose(ref_start, max(0, end - 0.5), abs_tol=1e-9, rel_tol=0)
            or not math.isclose(ref_end, min(limit, end + 2), abs_tol=1e-9, rel_tol=0)):
        raise ValueError("Turn or listener-reference window is inconsistent with the master clock")
    return start, end, ref_start, ref_end, limit


def resolve_sources(row):
    cid, speaker, listener = (row[key] for key in ("conversation_id", "speaker_id", "listener_id"))
    if speaker == listener or not all(isinstance(x, str) and SOURCE_ID.fullmatch(x) for x in (cid, speaker, listener)):
        raise ValueError("Invalid conversation or participant identifiers")
    folder = ROOT / "clean_sessions" / cid
    files = {role: {ext: folder / f"{cid}_{pid}.{ext}" for ext in ("wav", "mp4", "json", "npz")}
             for role, pid in (("speaker", speaker), ("listener", listener))}
    return folder, files


def crop_signature(row, sources):
    fingerprints = {}
    for role, files in sources.items():
        for modality in ("wav", "mp4"):
            p = files[modality]
            st = p.stat()
            fingerprints[f"{role}_{modality}"] = [str(p.relative_to(ROOT)), st.st_size, st.st_mtime_ns]
    data = {"version": CROP_VERSION, "row": row, "source_media": fingerprints,
            "crf": CRF, "preset": PRESET, "wav_format": "pcm_f32le"}
    return hashlib.sha256(json.dumps(data, allow_nan=False, sort_keys=True).encode()).hexdigest()


def load_existing(folder, signature):
    if not folder.is_dir():
        return None, False
    # Refuse to follow symlinks in an output directory even during resume.
    if any((folder / name).is_symlink() for name in OUTPUTS):
        raise ValueError(f"Symbolic link among existing crop outputs: {folder}")
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, False
    if not isinstance(meta, dict):
        return None, False
    preserve_l3 = meta.get("l3_profile") is not None
    try:
        mapping = meta["crop_mapping"]
        sizes = mapping["media_size_bytes"]
        verified = (set(meta) == META_BLOCKS and mapping["resume_signature"] == signature
                    and all((folder / name).is_file() and 0 < (folder / name).stat().st_size == sizes[name]
                            for name in OUTPUTS[:-1]))
        return (meta if verified else None), preserve_l3
    except (KeyError, TypeError, OSError):
        return None, preserve_l3


def build_meta(row, folder, source_dir, sources, signature, windows, speaker_audio, listener_audio,
               speaker_video, listener_video, sizes):
    start, end, ref_start, ref_end, duration = windows
    rel = lambda path: os.path.relpath(path, folder)
    return {
        "model_inputs": {"speaker_video": "speaker_A.mp4", "speaker_audio": "speaker_A.wav",
                         "transcript": row["text"]},
        "audit_refs": {
            "listener_ref_video": "listener_B_ref.mp4",
            "prev_other_turn_id": row.get("prev_other_turn_id"), "prev_other_text": row.get("prev_other_text"),
            "prev_other_gap_sec": row.get("prev_other_gap_sec"), "source_paths_relative_to": "meta.json directory",
            "source_session_dir": rel(source_dir),
            "source_files": {role: {ext: rel(path) for ext, path in files.items()}
                             for role, files in sources.items()},
        },
        "source_provenance": {key: row[key] for key in PROVENANCE_KEYS},
        "crop_mapping": {
            "master_clock": "standalone WAV, seconds relative to source audio sample 0",
            "raw_t_start": start, "raw_t_end": end, "ref_b_start": ref_start, "ref_b_end": ref_end,
            "time_axis_end_sec": duration,
            "raw_transcript_start": row.get("raw_transcript_start"),
            "raw_transcript_end": row.get("raw_transcript_end"),
            "speaker_wav": speaker_audio, "listener_preview_wav_slice": listener_audio,
            "speaker_mp4": speaker_video, "listener_ref_mp4": listener_video,
            "preview_audio_note": "MP4 AAC uses the synchronous standalone WAV slice, not original MP4 audio. Use speaker_A.wav for training and acoustic analysis.",
            "media_size_bytes": sizes, "resume_signature": signature,
        },
        "triage": {"triage_status": row["triage_status"], "reason_codes": row["reason_codes"],
                   "pilot_stratum": row.get("pilot_stratum"), "pilot_fallback_for": row.get("pilot_fallback_for"),
                   **{key: row.get(key) for key in TRIAGE_METRICS}},
        "l3_profile": None,
    }


def index_row(row, folder, *, elapsed, status, meta=None, error=None):
    mapping = meta["crop_mapping"] if meta else {}
    audio = mapping.get("speaker_wav", {})
    video = mapping.get("speaker_mp4", {})
    listener = mapping.get("listener_ref_mp4", {})
    paths = {name: (folder / name).relative_to(ROOT).as_posix() for name in OUTPUTS} if meta else {}
    sizes = {name: (folder / name).stat().st_size for name in OUTPUTS} if meta else {}
    return {
        **{key: row.get(key) for key in PROVENANCE_KEYS},
        "pilot_stratum": row.get("pilot_stratum"), "pilot_fallback_for": row.get("pilot_fallback_for"),
        "triage_status": row.get("triage_status"), "duration_bucket": row.get("duration_bucket"),
        "crop_status": status, "error": error, "elapsed_sec": elapsed,
        "speaker_wav_path": paths.get("speaker_A.wav"), "speaker_video_path": paths.get("speaker_A.mp4"),
        "listener_ref_video_path": paths.get("listener_B_ref.mp4"), "meta_json_path": paths.get("meta.json"),
        "speaker_wav_size_bytes": sizes.get("speaker_A.wav"), "speaker_video_size_bytes": sizes.get("speaker_A.mp4"),
        "listener_ref_video_size_bytes": sizes.get("listener_B_ref.mp4"), "meta_json_size_bytes": sizes.get("meta.json"),
        "total_bytes": sum(sizes.values()) if meta else None,
        "speaker_wav_duration_sec": audio.get("duration_sec"),
        "speaker_video_duration_sec": video.get("video_duration_sec"),
        "speaker_mp4_container_duration_sec": video.get("container_duration_sec"),
        "speaker_video_wav_abs_duration_diff_sec": video.get("video_wav_abs_duration_diff_sec"),
        "speaker_mp4_container_wav_abs_duration_diff_sec": video.get("container_wav_abs_duration_diff_sec"),
        "speaker_video_minus_audio_local_zero_master_sec": video.get("video_minus_audio_local_zero_master_sec"),
        "speaker_video_start_pts": video.get("video_start_pts"), "speaker_video_start_sec": video.get("video_start_sec"),
        "speaker_video_frame_count": video.get("video_frame_count"),
        "listener_ref_video_duration_sec": listener.get("video_duration_sec"),
        "listener_ref_video_start_pts": listener.get("video_start_pts"), "listener_ref_video_start_sec": listener.get("video_start_sec"),
        "listener_ref_video_frame_count": listener.get("video_frame_count"),
        "listener_ref_video_wav_abs_duration_diff_sec": listener.get("video_wav_abs_duration_diff_sec"),
        "listener_video_minus_audio_local_zero_master_sec": listener.get("video_minus_audio_local_zero_master_sec"),
    }


def crop_turn(row, output_root, ffmpeg, ffprobe):
    start_time = time.perf_counter()
    folder = output_root / row["turn_id"]
    try:
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ValueError(f"Unsafe crop output directory: {folder}")
        windows = master_windows(row)
        start, end, ref_start, ref_end, _ = windows
        source_dir, sources = resolve_sources(row)
        signature = crop_signature(row, sources)
        prior, has_l3 = load_existing(folder, signature)
        if prior:
            return index_row(row, folder, elapsed=time.perf_counter() - start_time, status="reused", meta=prior)
        if has_l3:
            raise ValueError("Refusing to replace a crop with a non-null l3_profile")
        speaker, listener = sources["speaker"], sources["listener"]
        with tempfile.TemporaryDirectory(prefix=f".{folder.name}.", dir=output_root) as work:
            stage = Path(work)
            a_wav, b_wav = stage / "speaker_A.wav", stage / "listener_temp.wav"
            a_mp4, b_mp4 = stage / "speaker_A.mp4", stage / "listener_B_ref.mp4"
            a_audio = cut_wav(speaker["wav"], a_wav, start, end)
            b_audio = cut_wav(listener["wav"], b_wav, ref_start, ref_end)
            a_video = cut_video(ffmpeg, ffprobe, speaker["mp4"], a_wav, a_mp4, start, end, a_audio,
                                probe_video(ffprobe, speaker["mp4"]))
            b_video = cut_video(ffmpeg, ffprobe, listener["mp4"], b_wav, b_mp4, ref_start, ref_end, b_audio,
                                probe_video(ffprobe, listener["mp4"]))
            b_wav.unlink()
            sizes = {name: (stage / name).stat().st_size for name in OUTPUTS[:-1]}
            meta = build_meta(row, folder, source_dir, sources, signature, windows, a_audio, b_audio,
                              a_video, b_video, sizes)
            with (stage / "meta.json").open("w", encoding="utf-8") as f:
                write_json(f, meta)
            if not folder.exists():
                stage.rename(folder)
            else:
                # For any incomplete earlier crop publish media first, metadata last.
                for name in OUTPUTS:
                    (stage / name).replace(folder / name)
        return index_row(row, folder, elapsed=time.perf_counter() - start_time, status="success", meta=meta)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        msg = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            msg += f": {exc.stderr.strip()[-2000:]}"
        return index_row(row, folder, elapsed=time.perf_counter() - start_time, status="failed", error=msg)


def distribution(values):
    return {"min": min(values) if values else None, "median": statistics.median(values) if values else None,
            "max": max(values) if values else None, "count": len(values)}


def summarize_crops(selected, rows, seconds, workers, full_pool, selection_path):
    good = [r for r in rows if r["crop_status"] != "failed"]
    fresh = [r for r in rows if r["crop_status"] == "success"]
    failed = [r for r in rows if r["crop_status"] == "failed"]
    output_bytes = sum(r["total_bytes"] for r in good)
    avg_new_wall = seconds / len(fresh) if fresh else None
    zero_checks = {key: all(abs(r[key]) <= PTS_ZERO_TOLERANCE for r in good) if good else None for key in
                   ("speaker_video_start_sec", "listener_ref_video_start_sec")}
    return {
        "selection_path": (selection_path.relative_to(ROOT).as_posix() if selection_path.is_relative_to(ROOT)
                           else str(selection_path)),
        "selected_count": len(selected), "success_count": len(good), "failed_count": len(failed),
        "newly_cropped_count": len(fresh), "reused_count": len(good) - len(fresh),
        "selected_strata": dict(Counter(r.get("pilot_stratum") or "unstratified" for r in selected)),
        "successful_strata": dict(Counter(r.get("pilot_stratum") or "unstratified" for r in good)),
        "status_by_duration_bucket": {status: dict(Counter(r["duration_bucket"] for r in selected
                                                            if r["triage_status"] == status))
                                      for status in sorted({r["triage_status"] for r in selected})},
        "fallback_from_pass_core": dict(Counter(r["pilot_fallback_for"] for r in selected if r.get("pilot_fallback_for"))),
        "selected_coverage": coverage(selected), "successful_coverage": coverage(good),
        "workers": workers, "wall_seconds": seconds, "wall_seconds_per_requested_turn": seconds / len(rows),
        "mean_worker_seconds_per_new_turn": statistics.mean(r["elapsed_sec"] for r in fresh) if fresh else None,
        "total_output_bytes": output_bytes, "total_output_mb": output_bytes / 1e6,
        "projection": {
            "full_pool_pass_review_count": full_pool,
            "wall_seconds": avg_new_wall * full_pool if avg_new_wall is not None else None,
            "output_bytes": round(output_bytes / len(good) * full_pool) if good else None,
            "output_gb": output_bytes / len(good) * full_pool / 1e9 if good else None,
            "note": "Linear estimate using the pilot throughput and mean payload size. Actual full-pool mix, CPU and storage may differ. Time estimate is null when no new files are cropped.",
        },
        "sync": {
            **{key: distribution([r[key] for r in good]) for key in
               ("speaker_video_wav_abs_duration_diff_sec", "speaker_mp4_container_wav_abs_duration_diff_sec",
                "speaker_video_minus_audio_local_zero_master_sec", "listener_ref_video_wav_abs_duration_diff_sec",
                "listener_video_minus_audio_local_zero_master_sec")},
            "all_video_starts_zero": zero_checks, "pts_zero_tolerance_sec": PTS_ZERO_TOLERANCE,
        },
        "failures": [{"turn_id": r["turn_id"], "error": r["error"]} for r in failed],
    }


def crop(metadata, selection_path, workers):
    started = time.perf_counter()
    selection_path = selection_path.resolve(strict=True)
    selected = pq.read_table(selection_path).to_pylist()
    validate_turn_ids(selected)
    output_root = output_directory(ROOT / "cropped_turns")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("crop requires both ffmpeg and ffprobe on PATH")
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(crop_turn, row, output_root, ffmpeg, ffprobe) for row in selected]
        for done, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results[r["turn_id"]] = r
            if r["crop_status"] == "failed":
                print(f"Failed {r['turn_id']}: {r['error']}", flush=True)
            if done % 20 == 0 or done == len(selected):
                print(f"Processed {done}/{len(selected)} crops", flush=True)
    rows = [results[row["turn_id"]] for row in selected]
    pool_file = metadata / "turns_candidates.parquet"
    full_pool = (sum(s in ("pass", "review") for s in pq.read_table(pool_file, columns=["triage_status"])
                     ["triage_status"].to_pylist()) if pool_file.is_file() else len(selected))
    report = summarize_crops(selected, rows, time.perf_counter() - started, workers, full_pool, selection_path)
    stem = selection_path.stem.removesuffix("_selection")
    write_table(metadata / f"{stem}_index", rows)
    atomic_write(metadata / f"step3_{stem}_crop_report.json", lambda file: write_json(file, report))
    print(f"Succeeded {report['success_count']}/{len(selected)} ({report['reused_count']} reused), "
          f"failed {report['failed_count']}; wall {report['wall_seconds']:.2f}s, {report['total_output_mb']:.2f} MB")
    return bool(report["failed_count"])


def positive_workers(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("--workers must be at least 1")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select-pilot", help="Select 180 diverse pass/review turns")
    render = commands.add_parser("crop", help="Crop a turn-selection Parquet, safely resuming completed outputs")
    render.add_argument("selection", type=Path, help="Path to a selection Parquet")
    render.add_argument("--workers", type=positive_workers, default=DEFAULT_WORKERS,
                        help=f"Concurrent FFmpeg crop jobs (default {DEFAULT_WORKERS})")
    args = parser.parse_args()
    try:
        metadata = output_directory(ROOT / "metadata")
        if args.command == "select-pilot":
            select_pilot(metadata)
        elif crop(metadata, args.selection, args.workers):
            raise SystemExit(1)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{args.command}: {exc}\n")


if __name__ == "__main__":
    main()
