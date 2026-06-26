#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import glob
import json
import os
import random
import shelve
from collections import defaultdict

import numpy as np
from decord import VideoReader, cpu
from filelock import FileLock

from eval.task.videoloader import VideoLoader

_video_loader: VideoLoader | None = None


def _get_video_loader() -> VideoLoader:
    global _video_loader
    if _video_loader is None:
        _video_loader = VideoLoader()
    return _video_loader


def get_rank_and_world():
    rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    if rank >= 0 and world >= 1:
        return rank, world
    return None, None


def make_sample_key(sample):
    return json.dumps(sample, sort_keys=True, ensure_ascii=False)


def db_load_keys(db_path, lock_path):
    jsonl_path = db_path + ".jsonl"
    if os.path.exists(jsonl_path):
        data = _load_jsonl(jsonl_path)
        return {
            k
            for k, v in data.items()
            if k != METADATA_KEY and not (isinstance(v, dict) and "error" in v)
        }
    with FileLock(lock_path):
        with shelve.open(db_path) as db:
            return {
                k for k, v in db.items() if not (isinstance(v, dict) and "error" in v)
            }


def db_flush(db_path, lock_path, cache):
    if not cache:
        return
    with FileLock(lock_path):
        with shelve.open(db_path, writeback=False) as db:
            for key, value in cache.items():
                db[key] = value


METADATA_KEY = "_metadata"


SHELVE_EXTS = (".db", ".dir", ".dat", ".bak", ".pag")


def db_maybe_export_jsonl(db_path, lock_path, expected_count):
    """Export shelve DB to JSONL if all samples are present, then remove shelve files.

    Condition: len(db) >= expected_count + 1  (samples + _metadata).
    """
    jsonl_path = db_path + ".jsonl"
    with FileLock(lock_path):
        with shelve.open(db_path) as db:
            if len(db) < expected_count + 1:
                return
            with open(jsonl_path, "w", encoding="utf-8") as f:
                metadata = db.get(METADATA_KEY)
                if metadata is not None:
                    f.write(
                        json.dumps({METADATA_KEY: metadata}, ensure_ascii=False) + "\n"
                    )
                for key, value in db.items():
                    if key == METADATA_KEY:
                        continue
                    f.write(
                        json.dumps({"_key": key, "_value": value}, ensure_ascii=False)
                        + "\n"
                    )
        # Remove shelve files
        for ext in ("", *SHELVE_EXTS):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)
    print(f"[jsonl] Exported {expected_count} entries to {jsonl_path}")


def _load_jsonl(jsonl_path):
    """Load a JSONL DB file into a dict matching shelve format."""
    result = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if METADATA_KEY in obj:
                result[METADATA_KEY] = obj[METADATA_KEY]
            else:
                result[obj["_key"]] = obj["_value"]
    return result


def db_load_all(db_path, lock_path):
    jsonl_path = db_path + ".jsonl"
    if os.path.exists(jsonl_path):
        return _load_jsonl(jsonl_path)
    with FileLock(lock_path):
        with shelve.open(db_path) as db:
            return dict(db)


def db_ensure_metadata(db_path, lock_path, metadata):
    with FileLock(lock_path):
        with shelve.open(db_path, writeback=False) as db:
            existing = db.get(METADATA_KEY)
            if existing is None:
                db[METADATA_KEY] = metadata
                return
            if existing != metadata:
                raise ValueError(
                    f"DB metadata mismatch at {db_path}.\n"
                    f"Existing: {json.dumps(existing, indent=2)}\n"
                    f"Current:  {json.dumps(metadata, indent=2)}"
                )


def load_video_frames(video_path, fps=1.0, start_sec=None, end_sec=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    avg_fps = vr.get_avg_fps()
    total = len(vr)
    frame_indices = np.arange(0, total, avg_fps / fps).round().astype(int)
    frame_indices = frame_indices.clip(0, total - 1)
    timestamps = frame_indices / avg_fps
    if start_sec is not None or end_sec is not None:
        mask = np.ones(len(timestamps), dtype=bool)
        if start_sec is not None:
            mask &= timestamps >= start_sec
        if end_sec is not None:
            mask &= timestamps <= end_sec
        frame_indices = frame_indices[mask]
        timestamps = timestamps[mask]
    del vr
    loader = _get_video_loader()
    frames = loader.run(video_path, frame_indices.tolist())
    return frames, timestamps


def load_annotations(anno_path=None, items=None):
    if items is not None:
        raw = items
    elif anno_path is not None:
        if anno_path.endswith(".json"):
            with open(anno_path) as f:
                raw = json.load(f)
        else:
            paths = sorted(glob.glob(os.path.join(anno_path, "*.json")))
            raw = []
            for p in paths:
                with open(p) as f:
                    raw.extend(json.load(f))
    else:
        raise ValueError("Either anno_path or items must be provided")

    groups = defaultdict(list)
    for item in raw:
        groups[item["video"]].append(item)

    video_keys = list(groups.keys())
    seed = os.getppid() if os.environ.get("LOCAL_RANK") else 42
    random.Random(seed).shuffle(video_keys)

    anno = []
    for key in video_keys:
        anno.extend(groups[key])
    return anno


def span_chunk(data, n_chunks, index):
    total = len(data)
    per_chunk = total // n_chunks
    remainder = total % n_chunks
    start = per_chunk * index + min(index, remainder)
    end = start + per_chunk + (1 if index < remainder else 0)
    return data[start:end]
