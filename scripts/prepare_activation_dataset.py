#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import argparse
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import cv2
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ANNOTATION_FILES = [
    "ActivityNet-Captions.jsonl",
    "Charades.jsonl",
    "DiDeMo.jsonl",
    "ET-Instruct.jsonl",
    "Grounded-VideoLLM.jsonl",
    "LITA.jsonl",
    "YouCook2.jsonl",
    # "shot2story.jsonl",
]

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".webm", ".m4v", ".mov", ".mpg", ".3gp"]

# ActivityNet-Captions has videos spread across multiple subdirectories
_ANET_CAPTIONS_SUBDIRS = [
    "v1-3/train_val",
    "v1-2/train",
    "v1-3/test",
    "v1-2/test",
    "v1-2/val",
    "missing_files",
]


def get_video_metadata(video_path):
    """Extracts FPS, duration, and resolution."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    return fps, duration, [height, width]


def _process_path(path):
    return path, get_video_metadata(path)


def _try_with_extensions(base_path):
    """Try base_path as-is, then with common video extensions."""
    if os.path.isfile(base_path):
        return base_path
    for ext in VIDEO_EXTENSIONS:
        candidate = base_path + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _search_anet_captions(video_root, video_id):
    """Search ActivityNet-Captions directories for a video_id."""
    base = os.path.join(video_root, "ActivityNet-Captions")
    for subdir in _ANET_CAPTIONS_SUBDIRS:
        result = _try_with_extensions(os.path.join(base, subdir, video_id))
        if result:
            return result
    return None


def resolve_video_path(video_root, source, metadata):
    """Resolve video path from annotation metadata.

    Directory structure after tar extraction:
      video_root/
        ActivityNet-Captions/  → v1-2/{train,test,val}/, v1-3/{train_val,test}/, missing_files/
        Charades/              → videos/
        CharadesEgo/           → videos/
        DiDeMo/                → raw/DiDeMo/LocalizingMoments/data/missing_videos/YFCC100M_videos/
        ET-Instruct-164K/      → {sub_source}/ (metadata.video has full relative path)
        activitynet/           → videos/
        coin/                  → videos/
        qvhighlights/          → videos/
        shot2story-videos/     → videos/
        youcook2/              → videos/
    """
    if source == "ActivityNet-Captions":
        return _search_anet_captions(video_root, metadata["video_id"])

    elif source in ("Charades", "CharadesEgo"):
        return _try_with_extensions(
            os.path.join(video_root, source, "videos", metadata["video_id"])
        )

    elif source == "DiDeMo":
        video_file = metadata["video"]
        didemo_base = os.path.join(
            video_root,
            "DiDeMo",
            "raw",
            "DiDeMo",
            "LocalizingMoments",
            "data",
            "missing_videos",
            "YFCC100M_videos",
        )
        path = os.path.join(didemo_base, video_file)
        if os.path.isfile(path):
            return path
        # Fallback: try stem with different extensions
        stem = os.path.splitext(video_file)[0]
        return _try_with_extensions(os.path.join(didemo_base, stem))

    elif source == "ET-Instruct-164K":
        # metadata.video has relative path: e.g., "how_to_step/PJi8ZEHAFcI.mp4"
        path = os.path.join(video_root, "ET-Instruct-164K", metadata["video"])
        if os.path.isfile(path):
            return path
        return None

    elif source == "Grounded-VideoLLM":
        vid = metadata["video_id"]
        meta_src = metadata.get("source", "")
        if meta_src == "anet":
            return _try_with_extensions(
                os.path.join(video_root, "activitynet", "videos", vid)
            )
        elif meta_src == "qvhighlights":
            return _try_with_extensions(
                os.path.join(video_root, "qvhighlights", "videos", vid)
            )
        return None

    elif source == "LITA":
        # LITA uses ActivityNet-Captions videos
        return _search_anet_captions(video_root, metadata["video_id"])

    elif source == "YouCook2":
        return _try_with_extensions(
            os.path.join(video_root, "youcook2", "videos", metadata["video"])
        )

    elif source == "shot2story":
        path = os.path.join(
            video_root, "shot2story-videos", "videos", metadata["video"]
        )
        if os.path.isfile(path):
            return path
        return None

    return None


def download_annotations():
    """Download all annotation files from HuggingFace."""
    paths = {}
    for name in ANNOTATION_FILES:
        path = hf_hub_download(
            repo_id="interlive/stream-data",
            filename=f"annotations/{name}",
            repo_type="dataset",
        )
        paths[name] = path
    return paths


def load_annotations(annotation_paths):
    """Load all annotation JSONL files."""
    items = []
    for name, path in annotation_paths.items():
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    items.append(item)
    return items


def convert(args):
    # Download and load annotations
    print("Downloading annotations from HuggingFace...")
    annotation_paths = download_annotations()

    print("Loading annotations...")
    items = load_annotations(annotation_paths)
    print(f"Loaded {len(items)} annotation items")

    # Resolve video paths and collect unique ones
    video_path_map = {}  # (source, video_key) -> resolved_path
    unique_paths = set()

    for item in tqdm(items, desc="Resolving paths", ncols=88, mininterval=1):
        source = item["video"]["source"]
        metadata = item["video"]["metadata"]
        video_key = metadata.get("video") or metadata.get("video_id", "")

        if (source, video_key) in video_path_map:
            continue

        resolved = resolve_video_path(args.video_root, source, metadata)
        if resolved:
            video_path_map[(source, video_key)] = resolved
            unique_paths.add(resolved)
        else:
            video_path_map[(source, video_key)] = None

    print(f"Found {len(unique_paths)} unique video files")

    # Extract metadata in parallel
    meta_cache = {}
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.imap_unordered(_process_path, list(unique_paths))
        for path, meta in tqdm(results, total=len(unique_paths), desc="Metadata"):
            if meta:
                meta_cache[path] = meta

    print(f"Valid metadata for {len(meta_cache)} videos")

    # Generate training samples
    stats = {
        "total": 0,
        "skipped_no_video": 0,
        "skipped_no_segment": 0,
        "skipped_bad_duration": 0,
        "skipped_seg_before_video": 0,
        "skipped_seg_after_duration": 0,
        "written": 0,
    }

    skip_details = defaultdict(lambda: defaultdict(int))  # {reason: {source: count}}

    with open(args.output_path, "w") as f_out:
        f_out.write(json.dumps({"_metadata": vars(args)}) + "\n")

        for item in tqdm(items, desc="Writing JSONL", ncols=88, mininterval=1):
            stats["total"] += 1

            source = item["video"]["source"]
            metadata = item["video"]["metadata"]
            video_key = metadata.get("video") or metadata.get("video_id", "")

            video_path = video_path_map.get((source, video_key))
            if video_path is None or video_path not in meta_cache:
                stats["skipped_no_video"] += 1
                skip_details["no_video"][source] += 1
                continue

            segments = item.get("segment", [])
            if not segments:
                stats["skipped_no_segment"] += 1
                skip_details["no_segment"][source] += 1
                continue

            fps, duration, resolution = meta_cache[video_path]
            video_start_time = item.get("start_time", 0) or 0
            video_end_time = item.get("end_time") or duration

            if abs(video_end_time - duration) >= 5:
                stats["skipped_bad_duration"] += 1
                skip_details["bad_duration"][source] += 1
                continue

            for seg_idx, segment in enumerate(segments):
                seg_start = segment["start_time"]
                seg_end = segment["end_time"]

                if seg_end <= seg_start or seg_end <= 0:
                    continue
                if seg_start < video_start_time - 5:
                    stats["skipped_seg_before_video"] += 1
                    skip_details["seg_before_video"][source] += 1
                    continue
                if seg_end > duration + 5:
                    stats["skipped_seg_after_duration"] += 1
                    skip_details["seg_after_duration"][source] += 1
                    continue

                # Compute U, L (segment-dependent, computed once)
                t_start, t_end = seg_start, seg_end
                t_dur = t_end - t_start

                U = t_start + max(
                    0,
                    args.inactive_curr_ratio * t_dur - args.inactive_curr_margin,
                )

                L = video_start_time
                for other in segments:
                    if other is segment:
                        continue
                    p_start = other["start_time"]
                    p_end = other["end_time"]
                    if p_end <= U:
                        candidate = min(
                            p_end,
                            p_start
                            + args.inactive_prev_ratio * (p_end - p_start)
                            + args.inactive_prev_margin,
                        )
                        L = max(L, candidate)

                # Duration constraints
                video_actual_dur = video_end_time - video_start_time
                dur_min = args.video_duration_min
                dur_max = min(args.video_duration_max, video_actual_dur)

                # 3 cut mode ranges (v_end bounds)
                after_limit = min(seg_end + args.context_after_max, video_end_time)
                cut_mode_ranges = [
                    (video_start_time, seg_start),
                    (seg_start, seg_end),
                    (seg_end, after_limit),
                ]
                vstart_lo = max(
                    video_start_time,
                    seg_start - args.context_before_max,
                )

                for cm_lo, cm_hi in cut_mode_ranges:
                    if cm_lo >= cm_hi:
                        continue

                    # Independent T_inactive, v_start per cut mode
                    # Retry up to 30 times if duration constraint fails
                    v_end = None
                    for _retry in range(30):
                        T_inactive = 0.0
                        apply_inactive = False
                        if U > 0:
                            bound = min(L, U)
                            if random.random() < args.inactive_minimal_ratio:
                                T_inactive = bound
                                apply_inactive = bound > vstart_lo
                            else:
                                T_inactive = random.uniform(bound, U)
                                apply_inactive = T_inactive > vstart_lo

                        if apply_inactive:
                            mode = random.choices(
                                ["min", "max", "uniform"],
                                weights=[
                                    args.vstart_minimal_ratio,
                                    args.vstart_maximal_ratio,
                                    1
                                    - args.vstart_minimal_ratio
                                    - args.vstart_maximal_ratio,
                                ],
                            )[0]
                            if mode == "min":
                                v_start = vstart_lo
                            elif mode == "max":
                                v_start = T_inactive
                                apply_inactive = False
                            else:
                                v_start = random.uniform(vstart_lo, T_inactive)
                        else:
                            v_start = random.uniform(vstart_lo, U)

                        # v_end: prefer [v_start+dur_min, v_start+dur_max] ∩ cut range
                        # If cut range is shorter than dur_min, use cm_hi as fallback
                        # context_after_inactive_min: hard constraint
                        inactive_end_bound = (
                            T_inactive + args.context_after_inactive_min
                            if apply_inactive
                            else 0.0
                        )
                        v_end_hi = min(cm_hi, v_start + dur_max)
                        v_end_lo = max(cm_lo, v_start + dur_min, inactive_end_bound)
                        if v_end_lo > v_end_hi:
                            # dur_min too large for this cut range → relax
                            v_end_lo = max(cm_lo, inactive_end_bound)
                        if v_end_lo <= v_end_hi and v_end_lo - v_start >= 1.0:
                            v_end = random.uniform(v_end_lo, v_end_hi)
                            break

                    if v_end is None:
                        continue

                    # Build sequence
                    sequence = []

                    # Previous segment description (if exists)
                    if args.include_prev_desc and seg_idx > 0:
                        prev_desc = segments[seg_idx - 1].get("description", "")
                        if prev_desc:
                            sequence.append(
                                {"type": "text", "content": prev_desc, "output": False}
                            )

                    # Query
                    query = item["query"]
                    if isinstance(query, list):
                        query = random.choice(query)
                    sequence.append({"type": "text", "content": query, "output": False})

                    # Video
                    sequence.append(
                        {
                            "type": "video",
                            "path": video_path,
                            "fps": fps,
                            "start_seconds": round(v_start, 3),
                            "end_seconds": round(v_end, 3),
                            "src_resolution": resolution,
                        }
                    )

                    # Activation output spec with entries
                    entries = []
                    if apply_inactive:
                        entries.append(
                            {
                                "start_seconds": round(v_start, 3),
                                "end_seconds": round(T_inactive, 3),
                                "mask": False,
                                "value": "<inactive>",
                            }
                        )
                    entries.append(
                        {
                            "start_seconds": round(seg_start, 3),
                            "end_seconds": round(seg_end, 3),
                            "mask": True,
                            "value": "<active>",
                        }
                    )
                    sequence.append({"type": "activation_output", "entries": entries})

                    f_out.write(json.dumps(sequence) + "\n")
                    stats["written"] += 1

    print("\nDone. Stats:")
    print(f"  Total items:          {stats['total']}")
    print(f"  Skipped (no video):   {stats['skipped_no_video']}")
    print(f"  Skipped (no seg):     {stats['skipped_no_segment']}")
    print(f"  Skipped (bad dur):    {stats['skipped_bad_duration']}")
    print(f"  Skipped (seg<start):  {stats['skipped_seg_before_video']}")
    print(f"  Skipped (seg>dur):    {stats['skipped_seg_after_duration']}")
    print(f"  Written samples:      {stats['written']}")
    if skip_details:
        print("\n  Skip breakdown by source:")
        for reason in sorted(skip_details):
            total = sum(skip_details[reason].values())
            print(f"    {reason}: {total}")
            for src, n in sorted(skip_details[reason].items(), key=lambda x: -x[1]):
                print(f"      {src}: {n}")
    print(f"\n  Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert annotations to MDM proactive trigger model training data"
    )
    parser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="Root directory for videos, organized as video_root/source/video_id",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--context_before_max",
        type=float,
        default=128.0,
        help="Max seconds of video context before segment",
    )
    parser.add_argument(
        "--context_after_max",
        type=float,
        default=128.0,
        help="Max seconds of video after segment end for 'after' cut mode",
    )
    parser.add_argument(
        "--include_prev_desc",
        action="store_true",
        default=False,
        help="Include previous segment description as context",
    )
    parser.add_argument(
        "--inactive_curr_ratio",
        type=float,
        default=0.2,
        help="Ratio of target segment duration for U computation",
    )
    parser.add_argument(
        "--inactive_curr_margin",
        type=float,
        default=2.0,
        help="Margin subtracted in U computation",
    )
    parser.add_argument(
        "--inactive_prev_ratio",
        type=float,
        default=0.8,
        help="Ratio of previous segment duration for L computation",
    )
    parser.add_argument(
        "--inactive_prev_margin",
        type=float,
        default=2.0,
        help="Margin added in L computation",
    )
    parser.add_argument(
        "--inactive_minimal_ratio",
        type=float,
        default=0.5,
        help="Probability of using min(L,U) as max_t vs uniform sampling",
    )
    parser.add_argument(
        "--vstart_minimal_ratio",
        type=float,
        default=0.2,
        help="Probability of fixing v_start to lo when max_t > 0",
    )
    parser.add_argument(
        "--vstart_maximal_ratio",
        type=float,
        default=0.1,
        help="Probability of fixing v_start to max_t when max_t > 0",
    )
    parser.add_argument(
        "--context_after_inactive_min",
        type=float,
        default=4.0,
        help="Min seconds of video after inactive region end",
    )
    parser.add_argument(
        "--video_duration_min",
        type=float,
        default=8.0,
        help="Preferred min video clip duration (best-effort, relaxed for short videos)",
    )
    parser.add_argument(
        "--video_duration_max",
        type=float,
        default=256.0,
        help="Max video clip duration in seconds",
    )

    args = parser.parse_args()
    convert(args)
