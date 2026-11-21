#!/usr/bin/env python3
"""Index and validate local Seamless Interaction improvised/train conversations."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

DATA_ROOT = Path("/home/jhl/dataset/seamless_interaction")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = Path("improvised/train")
MODALITIES = ("mp4", "wav", "json", "npz")
NAME = re.compile(r"^(V[^_]+_S[^_]+_I[^_]+)_(P[^.]+)\.(mp4|wav|json|npz)$")
MAX_DURATION_DIFF_S = 0.5

GROUP_FIELDS = (
    "conversation_id", "vendor_id", "session_id", "interaction_id",
    "participant_a", "participant_b", "participant_count", "status", "reasons",
    "official_vendor_id", "official_session_id", "official_participant_a",
    "official_participant_b", "relationship", "relationship_detail", "prompt_hash",
    "prompt_id_unique", "interaction_type",
)
MEMBER_FIELDS = (
    "participant_id", "participant_role", "modality", "tar_path", "member_name",
    "member_offset", "size_bytes",
)
MANIFEST_FIELDS = GROUP_FIELDS + (
    "scan_status", "session_dir", "expected_bytes", "review_reasons",
    "a_wav_duration_s", "a_wav_sample_rate_hz", "a_mp4_duration_s",
    "b_wav_duration_s", "b_wav_sample_rate_hz", "b_mp4_duration_s",
    "wav_duration_diff_s", "mp4_duration_diff_s", "max_duration_diff_s",
)


def publish(path, writer, *, binary=False):
    """Publish project outputs atomically; never open a destination for overwriting."""
    options = {"mode": "w+b"} if binary else {"mode": "w", "encoding": "utf-8", "newline": ""}
    temp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, **options)
    staged = Path(temp.name)
    try:
        with temp:
            writer(temp)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def write_csv(path, fields, rows):
    def write(file):
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    publish(path, write)


def write_json(path, data):
    publish(path, lambda file: file.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n"))


def list_files(root, suffix):
    def on_error(error):
        raise error

    return sorted(Path(directory) / name
                  for directory, _, names in os.walk(root, onerror=on_error)
                  for name in names if Path(name).suffix.lower() == suffix)


def parse_name(member_name):
    # Match the basename, but retain the exact TAR member path for extraction.
    match = NAME.fullmatch(PurePosixPath(member_name).name)
    return match.groups() if match else None


CSV_SCHEMAS = {
    "relationships.csv": (("vendor_id", "V"), ("session_id", "S")),
    "participants.csv": (("vendor_id", "V"), ("participant_id", "P")),
    "interactions.csv": (("prompt_hash", ""),),
}


def read_official_metadata(data_root):
    def key_for(row, columns):
        values = tuple((row.get(col) or "").strip() for col, _ in columns)
        if not all(values):
            return None
        return tuple(prefix + value.removeprefix(prefix) if prefix else value
                     for value, (_, prefix) in zip(values, columns))

    files = list_files(data_root, ".csv")
    tables, notes = {}, []
    for name, columns in CSV_SCHEMAS.items():
        paths = [p for p in files if p.name.lower() == name]
        table, conflicts = {}, set()
        for path in paths:
            try:
                with path.open(encoding="utf-8-sig", newline="") as source:
                    reader = csv.DictReader(source)
                    required = {column for column, _ in columns}
                    if name == "interactions.csv":
                        required.add("prompt_id_unique")
                    if not required.issubset(reader.fieldnames or ()):
                        notes.append(f"{path.relative_to(data_root)}: missing expected CSV columns")
                        continue
                    for row in reader:
                        key = key_for(row, columns)
                        if key is None or key in conflicts:
                            continue
                        if key in table and table[key] != row:
                            conflicts.add(key)
                            del table[key]
                        else:
                            table[key] = row
            except (OSError, ValueError, csv.Error) as error:
                # Do not publish joins from a partially readable table.
                notes.append(f"{name}: unreadable CSV; joins disabled: {error}")
                table.clear()
                break
        if not paths:
            notes.append(f"{name}: not found; corresponding official metadata fields are blank")
        if conflicts:
            notes.append(f"{name}: {len(conflicts)} conflicting keys omitted")
        tables[name] = table
    if not files:
        notes.append("No official CSV files found. Filename-derived IDs are still available.")
    return tables, {
        "csv_files_found": [p.relative_to(data_root).as_posix() for p in files],
        "tables": {name: {"joinable_keys": len(table), "matched": 0, "total": 0}
                   for name, table in tables.items()},
        "join_keys": {
            "relationships.csv": ["vendor_id", "session_id"],
            "participants.csv": ["vendor_id", "participant_id"],
            "interactions.csv": ["interaction_id without I prefix = prompt_hash"],
        },
        "notes": notes,
    }


def classify(people):
    reasons = []
    if len(people) == 1:
        reasons.append("single_participant")
    elif len(people) > 2:
        reasons.append("more_than_two_participants")
    missing, duplicates = {}, {}
    for participant, modalities in sorted(people.items()):
        if absent := [modality for modality in MODALITIES if modality not in modalities]:
            missing[participant] = absent
        if repeated := [modality for modality, members in sorted(modalities.items()) if len(members) > 1]:
            duplicates[participant] = repeated
    if missing:
        reasons.append("missing_modalities")
    if duplicates:
        reasons.append("duplicate_modalities")
    return reasons, missing, duplicates


def collect_members(data_root, tar_paths):
    groups, anomalies = {}, []
    for path in tar_paths:
        tar_path = path.relative_to(data_root).as_posix()
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                parsed = parse_name(member.name)
                if not member.isfile() or not parsed:
                    anomalies.append({"tar_path": tar_path, "member_name": member.name,
                                      "reason": "not_regular_file" if not member.isfile()
                                      else "unrecognized_filename"})
                    continue
                conversation, participant, modality = parsed
                participant_entries = groups.setdefault(conversation, {}).setdefault(participant, {})
                participant_entries.setdefault(modality, []).append({
                    "participant_id": participant,
                    "modality": modality,
                    "tar_path": tar_path,
                    "member_name": member.name,
                    "member_offset": member.offset,
                    "size_bytes": member.size,
                })
    return groups, anomalies


def scan(data_root, metadata):
    archive_dir = data_root / SPLIT
    if not archive_dir.is_dir():
        raise ValueError(f"TAR directory not found: {archive_dir}")
    tar_paths = list_files(archive_dir, ".tar")
    if not tar_paths:
        raise ValueError(f"No .tar files found under {archive_dir}")
    # No pairing decision is made until every TAR has been scanned successfully.
    groups, anomalies = collect_members(data_root, tar_paths)
    tables, official = read_official_metadata(data_root)

    def lookup(name, key):
        record = tables[name].get(key)
        stats = official["tables"][name]
        stats["total"] += 1
        stats["matched"] += record is not None
        return record or {}

    rows, incomplete = [], []
    reasons_count, status_count = Counter(), Counter()
    session_keys, participant_keys, prompt_ids = set(), set(), set()
    paired_cross_tar = single_person_entries = extraction_bytes = 0
    for conversation, people in sorted(groups.items()):
        vendor, session, interaction = conversation.split("_")
        people_sorted = sorted(people)
        single_person_entries += len(people)
        session_keys.add((vendor, session))
        participant_keys.update((vendor, p) for p in people)
        a = people_sorted[0] if len(people_sorted) <= 2 else ""
        b = people_sorted[1] if len(people_sorted) == 2 else ""
        roles = {participant: role for role, participant in (("A", a), ("B", b)) if participant}
        session_row = lookup("relationships.csv", (vendor, session))
        participant_rows = {p: lookup("participants.csv", (vendor, p)) for p in people_sorted}
        prompt_row = lookup("interactions.csv", (interaction[1:],))
        if prompt_id := prompt_row.get("prompt_id_unique", ""):
            prompt_ids.add(prompt_id)
        reasons, missing, duplicates = classify(people)
        status = "incomplete" if reasons else "paired_complete"
        status_count[status] += 1
        group = dict(zip(GROUP_FIELDS, (
            conversation, vendor, session, interaction, a, b, len(people), status, ";".join(reasons),
            session_row.get("vendor_id", ""), session_row.get("session_id", ""),
            participant_rows.get(a, {}).get("participant_id", ""),
            participant_rows.get(b, {}).get("participant_id", ""),
            session_row.get("relationship", ""), session_row.get("relationship_detail", ""),
            prompt_row.get("prompt_hash", ""), prompt_id, prompt_row.get("interaction_type", ""),
        )))
        group_rows = []
        for participant, modalities in sorted(people.items()):
            for _, entries in sorted(modalities.items()):
                group_rows.extend({**group, **entry, "participant_role": roles.get(participant, "")}
                                  for entry in entries)
        if reasons:
            reasons_count.update(reasons)
            incomplete.append({"conversation_id": conversation, "participants": people_sorted,
                               "reasons": reasons, "missing_modalities": missing,
                               "duplicate_modalities": duplicates})
        else:
            extraction_bytes += sum(row["size_bytes"] for row in group_rows)
            paired_cross_tar += len({row["tar_path"] for row in group_rows}) > 1
        rows.extend(group_rows)

    for name, stats in official["tables"].items():
        matched, total = stats["matched"], stats["total"]
        stats["status"] = ("complete" if matched and matched == total else
                           "partial" if matched else "not_joined")
        stats["counting_unit"] = "person entry" if name == "participants.csv" else "conversation"
    summary = {
        "archive_dir": str(archive_dir),
        "tar_count": len(tar_paths),
        "indexed_tar_members": len(rows),
        "single_person_entries": single_person_entries,
        "conversation_count": len(groups),
        "paired_complete": status_count["paired_complete"],
        "paired_complete_cross_tar": paired_cross_tar,
        "incomplete": status_count["incomplete"],
        "incomplete_reason_counts": dict(sorted(reasons_count.items())),
        "estimated_extract_bytes": extraction_bytes,
        "estimated_extract_gb": extraction_bytes / 10**9,
        "distinct_sessions": len(session_keys),
        "distinct_participants": len(participant_keys),
        "distinct_prompts": len(prompt_ids),
        "counting_notes": {
            "sessions": "Distinct (vendor_id, session_id), across all indexed entries.",
            "participants": "Distinct (vendor_id, participant_id), across all indexed entries.",
            "prompts": "Distinct matched nonempty prompt_id_unique values, not filename I tokens.",
            "participant_roles": "Local A/B labels sorted by participant ID, not official prompt roles.",
            "incomplete_reasons": "A conversation can have multiple reasons.",
        },
        "official_metadata": official,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "incomplete_details": incomplete,
    }
    write_csv(metadata / "tar_index.csv", GROUP_FIELDS + MEMBER_FIELDS, rows)
    write_json(metadata / "tar_scan_summary.json", summary)
    print(f"TARs: {len(tar_paths)} | indexed members: {len(rows)} | person entries: {single_person_entries}")
    print(f"Paired complete: {status_count['paired_complete']} (across TARs: {paired_cross_tar})")
    print(f"Incomplete: {status_count['incomplete']} | reasons: {dict(sorted(reasons_count.items()))}")
    print(f"Estimated extraction: {extraction_bytes / 10**9:.6f} GB ({extraction_bytes:,} bytes)")
    print(f"Distinct Session / Participant / Prompt: {len(session_keys)} / {len(participant_keys)} / {len(prompt_ids)}")
    print(f"CSV files: {', '.join(official['csv_files_found']) or '(none)'}")
    for name, stats in official["tables"].items():
        print(f"  {name}: {stats['status']} ({stats['matched']}/{stats['total']} {stats['counting_unit']} matches)")
    for note in official["notes"]:
        print(f"  Note: {note}")
    print(f"Anomalous TAR members: {len(anomalies)}")
    print(f"Wrote {metadata / 'tar_index.csv'} and {metadata / 'tar_scan_summary.json'}")


def load_pairs(index_path):
    groups = defaultdict(list)
    with index_path.open(encoding="utf-8", newline="") as index:
        for row in csv.DictReader(index):
            if row["status"] != "paired_complete":
                continue
            parsed = parse_name(row["member_name"])
            if parsed != (row["conversation_id"], row["participant_id"], row["modality"]):
                raise ValueError(f"Inconsistent member in scan index: {row['member_name']}")
            row["size_bytes"] = int(row["size_bytes"])
            row["member_offset"] = int(row["member_offset"])
            groups[row["conversation_id"]].append(row)
    expected = {(role, modality) for role in ("A", "B") for modality in MODALITIES}
    for conversation, members in groups.items():
        a, b = members[0]["participant_a"], members[0]["participant_b"]
        pairs = {(row["participant_role"], row["modality"]) for row in members}
        if (len(members) != len(expected) or pairs != expected or not a or not b or a == b
                or any(row["participant_id"] != {"A": a, "B": b}[row["participant_role"]]
                       for row in members)):
            raise ValueError(f"Expected exactly eight distinct A/B files in scan index: {conversation}")
    return groups


def copy_members(data_root, by_tar, errors, counters):
    archive_dir = (data_root / SPLIT).resolve()
    for relative, targets in sorted(by_tar.items()):
        path = (data_root / relative).resolve()
        if not path.is_relative_to(archive_dir) or path.suffix.lower() != ".tar":
            raise ValueError(f"TAR path outside expected archive directory: {relative}")
        remaining = {row["member_offset"]: (row, destination) for row, destination in targets}
        if len(remaining) != len(targets):
            raise ValueError(f"Duplicate TAR member offset in scan index: {relative}")
        try:
            with tarfile.open(path, "r:*") as archive:
                for member in archive:
                    if member.offset not in remaining:
                        continue
                    row, destination = remaining.pop(member.offset)
                    try:
                        if (not member.isfile() or member.name != row["member_name"]
                                or member.size != row["size_bytes"]):
                            raise ValueError("TAR header differs from scan index; rerun scan")
                        if destination.is_symlink():
                            raise ValueError("output is a symbolic link")
                        if (member.size > 0 and destination.is_file()
                                and destination.stat().st_size == member.size):
                            counters["skipped_same_size"] += 1
                        else:
                            def copy(staged):
                                with archive.extractfile(member) as source:
                                    shutil.copyfileobj(source, staged, length=1024 * 1024)
                                if staged.tell() != member.size:
                                    raise ValueError("extracted byte count differs from TAR header")
                            publish(destination, copy, binary=True)
                            counters["written"] += 1
                    except (OSError, ValueError, tarfile.TarError) as error:
                        errors[row["conversation_id"]].append(f"extract_error:{destination.name}: {error}")
                    if not remaining:
                        break
        except (OSError, tarfile.TarError) as error:
            for row, destination in remaining.values():
                errors[row["conversation_id"]].append(f"tar_error:{destination.name}: {error}")
        else:
            for row, destination in remaining.values():
                errors[row["conversation_id"]].append(f"missing_tar_member:{destination.name}: rerun scan")


def probe_stream(ffprobe, path, selector, entries):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", selector, "-show_entries",
         entries, "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode or result.stderr.strip():
        raise ValueError(result.stderr.strip() or f"ffprobe exited {result.returncode}")
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError(f"no stream matching {selector}")
    return info["streams"][0], info.get("format", {})


def positive_duration(*candidates):
    for candidate in candidates:
        if candidate not in (None, "", "N/A"):
            duration = float(candidate)
            if math.isfinite(duration) and duration > 0:
                return duration
    raise ValueError("no positive duration")


def video_duration(ffprobe, path):
    stream, container = probe_stream(ffprobe, path, "v:0", "stream=duration:format=duration")
    return positive_duration(stream.get("duration"), container.get("duration"))


def wav_info(ffprobe, path):
    try:
        with wave.open(str(path), "rb") as audio:
            rate, frames = audio.getframerate(), audio.getnframes()
            width = audio.getsampwidth() * audio.getnchannels()
            if rate <= 0 or frames <= 0 or width <= 0:
                raise ValueError("empty WAV or invalid WAV parameters")
            audio.setpos(frames - 1)
            if len(audio.readframes(1)) != width:
                raise ValueError("truncated WAV data")
            return rate, frames / rate
    except wave.Error as error:
        if not str(error).startswith("unknown format:"):
            raise
        # The corpus uses IEEE float WAVs, which Python's wave module cannot read.
        with path.open("rb") as audio:
            header = audio.read(12)
        if header[:4] == b"RIFF" and int.from_bytes(header[4:8], "little") + 8 > path.stat().st_size:
            raise ValueError("truncated WAV container") from error
        stream, container = probe_stream(
            ffprobe, path, "a:0", "stream=codec_name,sample_rate,duration:format=format_name",
        )
        if container.get("format_name") != "wav" or not stream.get("codec_name", "").startswith("pcm_"):
            raise ValueError("expected a PCM or IEEE float WAV stream") from error
        rate = int(stream.get("sample_rate", 0))
        if rate <= 0:
            raise ValueError("no positive WAV sample rate") from error
        return rate, positive_duration(stream.get("duration"))


def validate_member(path, row, ffprobe, np, metrics):
    if (path.is_symlink() or not path.is_file() or row["size_bytes"] <= 0
            or path.stat().st_size != row["size_bytes"]):
        raise ValueError("missing, empty, not a regular output file, or size differs from TAR header")
    role = row["participant_role"].lower()
    if row["modality"] == "json":
        with path.open(encoding="utf-8-sig") as source:
            json.load(source)
    elif row["modality"] == "npz":
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError("NPZ has no readable keys")
    elif row["modality"] == "wav":
        rate, duration = wav_info(ffprobe, path)
        metrics[f"{role}_wav_sample_rate_hz"] = rate
        metrics[f"{role}_wav_duration_s"] = duration
    else:
        metrics[f"{role}_mp4_duration_s"] = video_duration(ffprobe, path)


def extract(data_root, metadata, clean):
    # Lazy imports: scan can run without NumPy, pandas, pyarrow, or ffprobe.
    import numpy as np
    import pandas as pd
    import pyarrow  # noqa: F401  # Check Parquet support before writing any session files.

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ValueError("ffprobe is required for extract")
    groups = load_pairs(metadata / "tar_index.csv")
    destinations, by_tar, errors = {}, defaultdict(list), defaultdict(list)
    counters, reason_counts = Counter(), Counter()
    for conversation, members in groups.items():
        folder = output_dir(clean / conversation, data_root)
        destinations[conversation] = []
        for row in members:
            path = folder / PurePosixPath(row["member_name"]).name
            by_tar[row["tar_path"]].append((row, path))
            destinations[conversation].append((row, path))
    copy_members(data_root, by_tar, errors, counters)

    manifest = []
    for conversation, members in sorted(destinations.items()):
        group = {field: members[0][0][field] for field in GROUP_FIELDS}
        group.update(scan_status=group["status"], participant_count=int(group["participant_count"]),
                     session_dir=f"clean_sessions/{conversation}",
                     expected_bytes=sum(row["size_bytes"] for row, _ in members))
        reasons = errors[conversation]
        for row, path in members:
            try:
                validate_member(path, row, ffprobe, np, group)
            except Exception as error:  # A broken file should not stop validation of other sessions.
                reasons.append(f"invalid_{row['modality']}:{path.name}: {error}")
        differences = []
        for modality in ("wav", "mp4"):
            a = group.get(f"a_{modality}_duration_s")
            b = group.get(f"b_{modality}_duration_s")
            if a is not None and b is not None:
                difference = abs(a - b)
                group[f"{modality}_duration_diff_s"] = difference
                differences.append(difference)
                if difference > MAX_DURATION_DIFF_S and not math.isclose(difference, MAX_DURATION_DIFF_S, abs_tol=1e-9, rel_tol=0):
                    reasons.append(f"{modality}_duration_mismatch:{difference:.6f}s")
        group.update(status="review" if reasons else "ok",
                     review_reasons=" | ".join(reasons), max_duration_diff_s=max(differences, default=None))
        reason_counts.update({reason.partition(":")[0] for reason in reasons})
        manifest.append(group)

    distribution = {}
    for field in ("wav_duration_diff_s", "mp4_duration_diff_s", "max_duration_diff_s"):
        values = [row[field] for row in manifest if row.get(field) is not None]
        distribution[field] = {
            "count": len(values), "min": min(values, default=None),
            "median": statistics.median(values) if values else None, "max": max(values, default=None),
        }
    report = {
        "sessions": len(manifest),
        "ok": sum(row["status"] == "ok" for row in manifest),
        "review": sum(row["status"] == "review" for row in manifest),
        "duration_diff_threshold_s": MAX_DURATION_DIFF_S,
        "files_written": counters["written"],
        "files_skipped_same_size": counters["skipped_same_size"],
        "review_reason_counts": dict(sorted(reason_counts.items())),
        "duration_diff_distribution_s": distribution,
        "review_details": [{"conversation_id": key, "reasons": value}
                           for key, value in sorted(errors.items()) if value],
    }
    write_csv(metadata / "manifest_sessions.csv", MANIFEST_FIELDS, manifest)
    frame = pd.DataFrame(manifest, columns=MANIFEST_FIELDS)
    publish(metadata / "manifest_sessions.parquet",
            lambda file: frame.to_parquet(file, engine="pyarrow", index=False), binary=True)
    write_json(metadata / "step1_extract_report.json", report)
    print(f"Validated {report['sessions']} conversations: {report['ok']} ok, {report['review']} review")


def output_dir(path, data_root):
    if path.is_symlink() or path.resolve().is_relative_to(data_root):
        raise ValueError(f"Unsafe output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("scan", "Index TARs without extracting or modifying source files"),
                            ("extract", "Extract and validate complete pairs from the scan index")):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("--data-root", type=Path, default=DATA_ROOT)
        command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        data_root = args.data_root.resolve(strict=True)
        project = args.project_root.resolve()
        if not data_root.is_dir():
            raise ValueError(f"Data root is not a directory: {data_root}")
        metadata = output_dir(project / "metadata", data_root)
        if args.command == "scan":
            scan(data_root, metadata)
        else:
            extract(data_root, metadata, output_dir(project / "clean_sessions", data_root))
    except (ImportError, OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"{args.command}: {error}\n")


if __name__ == "__main__":
    main()
