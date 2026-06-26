#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import copy
import glob as glob_module
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from transformers import HfArgumentParser

from eval.task.utils import db_load_all, load_annotations, make_sample_key

from .compute_metrics import (
    dvc_eval,
    dvc_format,
    evs_eval,
    gvq_eval,
    tal_eval,
    tem_eval,
    tvg_eval,
    vhd_eval,
)

TRIGGER_TEMPORAL_TASKS = {"tvg", "epm", "tal", "tem", "evs", "vhd", "dvc", "slc"}
QA_REQUIRED_TASKS = {"gvq", "rar", "eca", "rvq"}


@dataclass
class ScoreArguments:
    anno_path: str = field(metadata={"help": "ETBench annotation JSON"})
    db_name: List[str] = field(
        metadata={"help": "Shelve DB path(s) — shell glob works without quotes"}
    )
    tasks: str = field(
        default="",
        metadata={
            "help": "Comma-separated tasks (empty = all). "
            "All: tvg,epm,tal,tem,evs,vhd,dvc,slc,gvq,rar,eca,rvq"
        },
    )


# ── Format helpers ────────────────────────────────────────────


def format_trigger_as_answer(segments, task):
    if not segments:
        return ""

    if task in ("tvg", "epm"):
        seg = segments[0]
        return f"{seg['start_sec']:.1f} - {seg['end_sec']:.1f} seconds"

    elif task in ("tal", "tem", "evs"):
        return "\n".join(
            f"{s['start_sec']:.1f} - {s['end_sec']:.1f} seconds" for s in segments
        )

    elif task in ("dvc", "slc"):
        return "\n".join(
            f"{s['start_sec']:.1f} - {s['end_sec']:.1f} seconds, detected event segment"
            for s in segments
        )

    elif task == "vhd":
        seg = segments[0]
        return f"{(seg['start_sec'] + seg['end_sec']) / 2:.1f}"

    return ""


# ── DVC temporal-only F1 (avoids PTBTokenizer) ───────────────


def dvc_temporal_f1_eval(samples):
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    cnt = 0
    gt_dict = {}
    pred_dict = {}

    for sample in samples:
        time, cap = dvc_format(sample["a"])
        if time is None or cap is None:
            cnt += 1
            continue
        gt_dict[sample["video"]] = {"timestamps": sample["tgt"]}
        pred_dict[sample["video"]] = [{"timestamp": t} for t in time]

    def _iou(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = min(max(a[1], b[1]) - min(a[0], b[0]), (a[1] - a[0]) + (b[1] - b[0]))
        return inter / (union + 1e-8)

    scale = len(pred_dict) / len(samples) if len(samples) > 0 else 0
    vid_ids = list(gt_dict.keys())
    f1_scores = []
    out = dict(Total=len(samples), Failed=cnt)

    for tiou in iou_thr:
        recalls, precisions = [], []
        for vid_id in vid_ids:
            refs = gt_dict[vid_id]
            ref_covered = set()
            pred_covered = set()
            if vid_id in pred_dict:
                for pi, p in enumerate(pred_dict[vid_id]):
                    for ri, ref_ts in enumerate(refs["timestamps"]):
                        if _iou(p["timestamp"], ref_ts) > tiou:
                            ref_covered.add(ri)
                            pred_covered.add(pi)
                precisions.append(len(pred_covered) / (pi + 1))
            else:
                precisions.append(0)
            recalls.append(len(ref_covered) / len(refs["timestamps"]))

        rec = sum(recalls) / len(recalls) * scale if recalls else 0
        prc = sum(precisions) / len(precisions) * scale if precisions else 0
        f1 = 0 if prc + rec == 0 else 2 * prc * rec / (prc + rec)
        out[f"F1@{tiou}"] = round(f1, 5)
        f1_scores.append(f1)

    out["F1"] = round(sum(f1_scores) / len(f1_scores), 5) if f1_scores else 0
    return out


def _dvc_eval_safe(samples, st):
    try:
        return dvc_eval(samples, st)
    except TypeError:
        return dvc_temporal_f1_eval(samples)


EVAL_DISPATCH = {
    "tvg": lambda samples, st: tvg_eval(samples),
    "epm": lambda samples, st: tvg_eval(samples),
    "tal": lambda samples, st: tal_eval(samples),
    "evs": lambda samples, st: evs_eval(samples),
    "vhd": lambda samples, st: vhd_eval(samples),
    "tem": lambda samples, st: tem_eval(samples),
    "gvq": lambda samples, st: gvq_eval(samples, st),
    "dvc": _dvc_eval_safe,
    "slc": _dvc_eval_safe,
}


# ── DB resolution ─────────────────────────────────────────────


DB_EXTS = (".jsonl", ".db", ".dir", ".dat", ".bak", ".pag")


def _db_exists(db_path):
    """Check if a DB (shelve or JSONL) exists at the given path."""
    for ext in ("", *DB_EXTS):
        if os.path.exists(db_path + ext):
            return True
    return False


def resolve_db_paths(db_names):
    """Resolve db_name list to unique, valid DB stems.

    Handles three cases:
      1) shell-expanded glob  → --db_name /path/db_*.db /path/db_*.jsonl  (strip ext, dedup)
      2) quoted glob          → --db_name '/path/db_*'  (internal expansion)
      3) explicit single path → --db_name /path/db_foo
    """
    stems = set()
    for name in db_names:
        # Internal glob expansion (for quoted patterns)
        if any(c in name for c in "*?["):
            for ext in DB_EXTS:
                for match in glob_module.glob(name + ext):
                    stems.add(match[: -len(ext)])
            continue

        # Strip known extension if present (shell-expanded paths)
        stem = name
        for ext in DB_EXTS:
            if name.endswith(ext):
                stem = name[: -len(ext)]
                break
        if _db_exists(stem):
            stems.add(stem)
    return sorted(stems)


# ── Evaluation core ───────────────────────────────────────────


def evaluate_db(db_path, anno, verbose=True):
    """Run evaluation on a single DB. Returns (collected, collected_est)."""
    lock_path = os.path.join(os.path.dirname(db_path), "file.lock")
    db = db_load_all(db_path, lock_path)
    db.pop("_metadata", None)

    if verbose:
        print(f"DB: {db_path} ({len(db)} samples)")

    predictions = []
    stats = defaultdict(lambda: {"matched": 0, "empty": 0, "total": 0})

    for sample in anno:
        task = sample["task"]
        if task not in TRIGGER_TEMPORAL_TASKS:
            continue

        st = stats[task]
        st["total"] += 1

        key = make_sample_key(sample)
        db_entry = db.get(key)

        pred = copy.deepcopy(sample)
        pred["_matched"] = db_entry is not None
        if db_entry:
            segments = db_entry.get("trigger_segments", [])
            for s in segments:
                s["start_sec"] = max(0.0, s["start_sec"])
                s["end_sec"] = max(0.0, s["end_sec"])
            pred["a"] = format_trigger_as_answer(segments, task)
            if pred["a"]:
                st["matched"] += 1
            else:
                pred["a"] = ""
                st["empty"] += 1
        else:
            pred["a"] = ""
            st["empty"] += 1

        predictions.append(pred)

    if verbose:
        print("\nCoverage:")
        for t in sorted(stats):
            s = stats[t]
            print(f"  {t}: {s['matched']}/{s['total']} matched, {s['empty']} empty")

    grouped = defaultdict(lambda: defaultdict(list))
    for sample in predictions:
        grouped[sample["task"]][sample["source"]].append(sample)

    collected = {}
    collected_est = {}
    for task in sorted(grouped):
        if task not in EVAL_DISPATCH:
            continue
        if task in QA_REQUIRED_TASKS:
            continue

        eval_fn = EVAL_DISPATCH[task]
        collected[task] = {}
        collected_est[task] = {}
        for source in sorted(grouped[task]):
            samples = grouped[task][source]
            matched = [s for s in samples if s.get("_matched")]
            try:
                collected[task][source] = eval_fn(samples, None)
            except Exception as e:
                if verbose:
                    print(f"\n  [{task}/{source}] Error: {e}")
                collected[task][source] = {"error": str(e)}
            if matched:
                try:
                    collected_est[task][source] = eval_fn(matched, None)
                except Exception:
                    pass

    return collected, collected_est, dict(stats)


def summarize_f1(collected):
    """Extract per-task average F1 (or mRec) across sources."""
    summary = {}
    for task, sources in collected.items():
        vals = []
        for d in sources.values():
            if "error" not in d:
                v = d.get("F1") or d.get("mRec")
                if isinstance(v, (int, float)):
                    vals.append(v)
        if vals:
            summary[task] = sum(vals) / len(vals)
    return summary


# ── Print helpers ─────────────────────────────────────────────


def print_section(title, headers, rows):
    if not rows:
        return
    print(f"\n{title}\n")
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in col_widths]))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def print_markdown_table(title, tasks, rows, progress=None):
    """Print a markdown summary table.

    rows: list of (db_path, {task: f1_value})
    progress: optional list of "matched/total" strings, one per row
    """
    if not rows:
        return
    headers = ["Method / Model"]
    if progress:
        headers.append("Progress")
    headers += [f"{t.upper()} (F1)" for t in tasks]
    print(f"\n{title}\n")
    print("| " + " | ".join(headers) + " |")
    sep = [":" + "-" * (len(headers[0]) - 1)]
    for h in headers[1:]:
        sep.append(":------:")
    print("| " + " | ".join(sep) + " |")
    for i, (db_path, summary) in enumerate(rows):
        cells = [db_path]
        if progress:
            cells.append(progress[i])
        for t in tasks:
            v = summary.get(t)
            cells.append(f"{v * 100:.1f}" if v is not None else "-")
        print("| " + " | ".join(cells) + " |")


def print_detailed_results(db_name, collected, collected_est):
    """Print detailed per-task, per-source tables (single DB mode)."""
    print("\n" + "=" * 60)
    print(f"Results: {db_name}")
    print("=" * 60)

    grounding_tasks = [t for t in ("tvg", "epm", "tal", "evs", "vhd") if t in collected]
    if grounding_tasks:
        headers = (
            "Task",
            "Source",
            "Total",
            "Done",
            "Failed",
            "F1@0.3",
            "F1@0.5",
            "F1@0.7",
            "F1",
            "Est.F1",
        )
        rows = []
        for task in grounding_tasks:
            for source in sorted(collected[task]):
                d = collected[task][source]
                if "error" in d:
                    continue
                est = collected_est.get(task, {}).get(source, {})
                rows.append(
                    (
                        task,
                        source,
                        d["Total"],
                        est.get("Total", "-"),
                        d["Failed"],
                        d.get("F1@0.3", "-"),
                        d.get("F1@0.5", "-"),
                        d.get("F1@0.7", "-"),
                        d.get("F1", "-"),
                        est.get("F1", "-"),
                    )
                )
        print_section("Grounding", headers, rows)

    caption_tasks = [t for t in ("dvc", "slc") if t in collected]
    if caption_tasks:
        headers = (
            "Task",
            "Source",
            "Total",
            "Done",
            "Failed",
            "F1@0.3",
            "F1@0.5",
            "F1@0.7",
            "F1",
            "Est.F1",
        )
        rows = []
        for task in caption_tasks:
            for source in sorted(collected[task]):
                d = collected[task][source]
                if "error" in d:
                    continue
                est = collected_est.get(task, {}).get(source, {})
                rows.append(
                    (
                        task,
                        source,
                        d["Total"],
                        est.get("Total", "-"),
                        d["Failed"],
                        d.get("F1@0.3", "-"),
                        d.get("F1@0.5", "-"),
                        d.get("F1@0.7", "-"),
                        d.get("F1", "-"),
                        est.get("F1", "-"),
                    )
                )
        print_section("Dense Captioning (temporal F1)", headers, rows)

    complex_tasks = [t for t in ("tem",) if t in collected]
    if complex_tasks:
        headers = (
            "Task",
            "Source",
            "Total",
            "Done",
            "Failed",
            "R@0.3",
            "R@0.5",
            "R@0.7",
            "mRec",
            "Est.mRec",
        )
        rows = []
        for task in complex_tasks:
            for source in sorted(collected[task]):
                d = collected[task][source]
                if "error" in d:
                    continue
                est = collected_est.get(task, {}).get(source, {})
                rows.append(
                    (
                        task,
                        source,
                        d.get("Total", "-"),
                        est.get("Total", "-"),
                        d.get("Failed", "-"),
                        d.get("R@0.3", "-"),
                        d.get("R@0.5", "-"),
                        d.get("R@0.7", "-"),
                        d.get("mRec", "-"),
                        est.get("mRec", "-"),
                    )
                )
        print_section("Complex", headers, rows)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary (avg F1 across sources)")
    print(f"{'=' * 60}")
    for task in sorted(collected):
        f1s = []
        est_f1s = []
        for source, d in collected[task].items():
            if "error" not in d:
                v = d.get("F1") or d.get("mRec")
                if isinstance(v, (int, float)):
                    f1s.append(v)
            est_d = collected_est.get(task, {}).get(source, {})
            if est_d and "error" not in est_d:
                ev = est_d.get("F1") or est_d.get("mRec")
                if isinstance(ev, (int, float)):
                    est_f1s.append(ev)
        if f1s:
            avg = sum(f1s) / len(f1s)
            est_avg = sum(est_f1s) / len(est_f1s) if est_f1s else 0
            print(f"  {task}: {avg * 100:.1f}  (est. {est_avg * 100:.1f})")


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = HfArgumentParser((ScoreArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    # Resolve DB paths (supports glob wildcards)
    db_paths = resolve_db_paths(args.db_name)
    if not db_paths:
        print(
            f"Error: No shelve DB found matching '{args.db_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load annotations once
    anno = load_annotations(args.anno_path)
    task_set = set(args.tasks.split(",")) if args.tasks else None
    if task_set:
        anno = [s for s in anno if s["task"] in task_set]
    print(f"Annotations: {len(anno)} samples")

    if len(db_paths) == 1:
        # Single DB — detailed output
        collected, collected_est, _ = evaluate_db(
            db_paths[0], anno, verbose=True
        )
        print_detailed_results(db_paths[0], collected, collected_est)
    else:
        # Multiple DBs — summary markdown tables only
        task_list = (
            args.tasks.split(",") if args.tasks else sorted(TRIGGER_TEMPORAL_TASKS)
        )
        eval_rows = []
        est_rows = []
        progress_list = []
        for db_path in db_paths:
            print(f"Evaluating: {db_path} ...", file=sys.stderr)
            collected, collected_est, stats = evaluate_db(
                db_path, anno, verbose=False
            )
            matched = sum(s["matched"] for s in stats.values())
            total = sum(s["total"] for s in stats.values())
            progress_list.append(f"{matched}/{total}")
            eval_rows.append((db_path, summarize_f1(collected)))
            est_rows.append((db_path, summarize_f1(collected_est)))

        print_markdown_table("evaluated:", task_list, eval_rows)
        print_markdown_table("estimate:", task_list, est_rows, progress=progress_list)


if __name__ == "__main__":
    main()
