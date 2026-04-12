from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Segment:
    file_name: str
    tid: int
    issue_type: str
    suggested_label: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    frames: int
    current_label_mode: str
    mean_conf: float
    mean_bbox_ar: float
    mean_down_vel: float
    fall_cue_rate: float
    fall_vel_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine hard-case segments from fall_debug_timeline JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timeline",
        action="append",
        default=[],
        help="Path to a timeline JSON file (repeatable).",
    )
    parser.add_argument(
        "--timeline_glob",
        default="",
        help="Glob pattern for timeline JSON files, e.g. runs/qt_outputs/fall_debug_timeline_cuda_*.json",
    )
    parser.add_argument("--out_csv", default="runs/qt_outputs/timeline_hardcases.csv")
    parser.add_argument("--out_summary", default="runs/qt_outputs/timeline_hardcases_summary.json")
    parser.add_argument("--min_segment_frames", type=int, default=4)
    parser.add_argument("--min_unknown_segment_frames", type=int, default=10)
    parser.add_argument("--max_frame_gap", type=int, default=2)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for raw in args.timeline:
        p = Path(raw).expanduser().resolve()
        if p.exists():
            files.append(p)
    if args.timeline_glob:
        files.extend(sorted(Path().glob(args.timeline_glob)))
    dedup: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup.append(rp)
    return dedup


def classify_issue(record: dict[str, object]) -> tuple[str, str] | None:
    label = str(record.get("label", "?"))
    fall_cue = bool(record.get("fall_cue", False))
    fall_vel = bool(record.get("fall_vel", False))
    bbox_ar = float(record.get("bbox_ar", 1.0))
    down_vel = float(record.get("down_vel", 0.0))
    conf = float(record.get("conf", 0.0))

    if label == "Fall":
        if not fall_cue and not fall_vel:
            if bbox_ar < 1.05:
                suggested = "Standing" if bbox_ar < 0.70 else "Walking"
                return ("false_fall_upright", suggested)
            if bbox_ar < 1.25:
                return ("false_fall_unclear", "Sitting")
        if conf < 0.42 and not (fall_cue or fall_vel):
            return ("low_conf_fall", "Walking")
        return None

    if label in {"Walking", "Standing", "Sitting", "?"}:
        strong_drop = down_vel > 0.13
        prone_shape = bbox_ar > 1.05
        if (fall_cue or fall_vel) and (strong_drop or prone_shape):
            return ("missed_fall_signal", "Fall")

    if label == "?":
        if bbox_ar < 0.78 and abs(down_vel) < 0.06:
            return ("long_unknown_upright", "Standing")
        if 0.78 <= bbox_ar <= 1.15 and abs(down_vel) < 0.10:
            return ("long_unknown_motion", "Walking")

    return None


def finalize_segment(
    file_name: str,
    tid: int,
    issue_type: str,
    suggested_label: str,
    records: list[dict[str, object]],
) -> Segment:
    labels = [str(r.get("label", "?")) for r in records]
    label_mode = Counter(labels).most_common(1)[0][0] if labels else "?"
    return Segment(
        file_name=file_name,
        tid=int(tid),
        issue_type=issue_type,
        suggested_label=suggested_label,
        start_frame=int(records[0]["frame"]),
        end_frame=int(records[-1]["frame"]),
        start_sec=float(records[0].get("sec", 0.0)),
        end_sec=float(records[-1].get("sec", 0.0)),
        frames=int(len(records)),
        current_label_mode=label_mode,
        mean_conf=float(sum(float(r.get("conf", 0.0)) for r in records) / max(len(records), 1)),
        mean_bbox_ar=float(sum(float(r.get("bbox_ar", 1.0)) for r in records) / max(len(records), 1)),
        mean_down_vel=float(sum(float(r.get("down_vel", 0.0)) for r in records) / max(len(records), 1)),
        fall_cue_rate=float(sum(1 for r in records if bool(r.get("fall_cue", False))) / max(len(records), 1)),
        fall_vel_rate=float(sum(1 for r in records if bool(r.get("fall_vel", False))) / max(len(records), 1)),
    )


def mine_segments(path: Path, args: argparse.Namespace) -> list[Segment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []

    mined: list[Segment] = []
    active: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    last_frame_by_key: dict[tuple[int, str, str], int] = {}

    for raw in records:
        if not isinstance(raw, dict):
            continue
        issue = classify_issue(raw)
        if issue is None:
            continue

        issue_type, suggested_label = issue
        tid = int(raw.get("tid", -1))
        frame = int(raw.get("frame", -1))
        key = (tid, issue_type, suggested_label)
        prev_frame = last_frame_by_key.get(key, -10_000)

        if key not in active or (frame - prev_frame) > args.max_frame_gap:
            if key in active and active[key]:
                segment = finalize_segment(
                    file_name=path.name,
                    tid=tid,
                    issue_type=issue_type,
                    suggested_label=suggested_label,
                    records=active[key],
                )
                min_len = (
                    args.min_unknown_segment_frames
                    if issue_type.startswith("long_unknown")
                    else args.min_segment_frames
                )
                if segment.frames >= min_len:
                    mined.append(segment)
            active[key] = []

        active[key].append(raw)
        last_frame_by_key[key] = frame

    for key, seg_records in active.items():
        if not seg_records:
            continue
        tid, issue_type, suggested_label = key
        segment = finalize_segment(
            file_name=path.name,
            tid=tid,
            issue_type=issue_type,
            suggested_label=suggested_label,
            records=seg_records,
        )
        min_len = args.min_unknown_segment_frames if issue_type.startswith("long_unknown") else args.min_segment_frames
        if segment.frames >= min_len:
            mined.append(segment)

    return mined


def write_csv(out_csv: Path, segments: Iterable[Segment]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "tid",
        "issue_type",
        "suggested_label",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "frames",
        "current_label_mode",
        "mean_conf",
        "mean_bbox_ar",
        "mean_down_vel",
        "fall_cue_rate",
        "fall_vel_rate",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in segments:
            writer.writerow(
                {
                    "file_name": seg.file_name,
                    "tid": seg.tid,
                    "issue_type": seg.issue_type,
                    "suggested_label": seg.suggested_label,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "start_sec": f"{seg.start_sec:.3f}",
                    "end_sec": f"{seg.end_sec:.3f}",
                    "frames": seg.frames,
                    "current_label_mode": seg.current_label_mode,
                    "mean_conf": f"{seg.mean_conf:.4f}",
                    "mean_bbox_ar": f"{seg.mean_bbox_ar:.4f}",
                    "mean_down_vel": f"{seg.mean_down_vel:.4f}",
                    "fall_cue_rate": f"{seg.fall_cue_rate:.4f}",
                    "fall_vel_rate": f"{seg.fall_vel_rate:.4f}",
                }
            )


def write_summary(out_summary: Path, segments: list[Segment], files: list[Path]) -> None:
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    by_issue: dict[str, int] = defaultdict(int)
    by_suggested: dict[str, int] = defaultdict(int)
    by_file: dict[str, int] = defaultdict(int)
    total_frames = 0
    for seg in segments:
        by_issue[seg.issue_type] += 1
        by_suggested[seg.suggested_label] += 1
        by_file[seg.file_name] += 1
        total_frames += seg.frames

    summary = {
        "input_files": [str(p) for p in files],
        "num_input_files": len(files),
        "num_segments": len(segments),
        "total_segment_frames": int(total_frames),
        "issue_distribution": dict(sorted(by_issue.items(), key=lambda item: item[0])),
        "suggested_label_distribution": dict(sorted(by_suggested.items(), key=lambda item: item[0])),
        "file_distribution": dict(sorted(by_file.items(), key=lambda item: item[0])),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = resolve_inputs(args)
    if not files:
        raise SystemExit("No timeline files found. Use --timeline or --timeline_glob.")

    all_segments: list[Segment] = []
    for path in files:
        all_segments.extend(mine_segments(path, args))

    out_csv = Path(args.out_csv).resolve()
    out_summary = Path(args.out_summary).resolve()
    write_csv(out_csv, all_segments)
    write_summary(out_summary, all_segments, files)

    print("=" * 72)
    print("TIMELINE HARDCASE MINING COMPLETE")
    print("=" * 72)
    print(f"Input files: {len(files)}")
    print(f"Segments: {len(all_segments)}")
    print(f"CSV: {out_csv}")
    print(f"Summary: {out_summary}")
    print("=" * 72)


if __name__ == "__main__":
    main()
