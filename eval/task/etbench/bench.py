#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import os
import traceback
from dataclasses import asdict, dataclass, field

import torch
from tqdm import tqdm
from transformers import HfArgumentParser

from eval.model import get_model_class
from eval.task.utils import (
    db_ensure_metadata,
    db_flush,
    db_load_keys,
    db_maybe_export_jsonl,
    get_rank_and_world,
    load_annotations,
    load_video_frames,
    make_sample_key,
    span_chunk,
)

MULTI_EVENT_TASKS = {"dvc", "slc", "evs", "vhd", "tal", "tem"}


@dataclass
class BenchArguments:
    model_type: str = field(
        metadata={"help": "Registry key (e.g. qwen3_stride)"}
    )
    model_path: str = field(metadata={"help": "Path to pretrained model checkpoint"})
    anno_path: str = field(metadata={"help": "ETBench annotation JSON or directory"})
    data_path: str = field(
        metadata={"help": "Root directory containing ETBench videos"}
    )
    pred_path: str = field(metadata={"help": "Output directory for shelve DB and lock"})
    db_name: str = field(metadata={"help": "Shelve DB name (e.g. trigger_stride)"})
    tasks: str = field(
        default="",
        metadata={
            "help": "Comma-separated tasks (empty = all). "
            "All: tvg,epm,tal,tem,evs,vhd,dvc,slc,gvq,rar,eca,rvq"
        },
    )
    chunk: int = field(default=1)
    index: int = field(default=0)
    verbose: bool = field(default=False)
    flush_every: int = field(default=8)
    event_mode: str = field(
        default="default",
        metadata={"help": "Event mode: default (use task), multi, single"},
    )


if __name__ == "__main__":
    parser1 = HfArgumentParser((BenchArguments,))
    bench_args, remaining = parser1.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    model_cls = get_model_class(bench_args.model_type)
    trigger_args_cls = model_cls.TRIGGER_ARGUMENTS

    parser2 = HfArgumentParser((trigger_args_cls,))
    (trigger_args,) = parser2.parse_args_into_dataclasses(remaining)

    rank, world = get_rank_and_world()
    if rank is not None:
        index, chunk = rank, world
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        index, chunk = bench_args.index, bench_args.chunk
        device = "cuda"

    os.makedirs(bench_args.pred_path, exist_ok=True)

    db_path = os.path.join(bench_args.pred_path, bench_args.db_name)
    lock_path = os.path.join(bench_args.pred_path, "file.lock")

    metadata = {
        "bench_args": asdict(bench_args),
        "trigger_args": asdict(trigger_args),
    }
    db_ensure_metadata(db_path, lock_path, metadata)

    print(f"[rank {index}/{chunk}] DB: {db_path}")
    print(f"Model: {bench_args.model_type} @ {bench_args.model_path}")

    anno = load_annotations(bench_args.anno_path)
    if bench_args.tasks:
        task_set = set(bench_args.tasks.split(","))
        anno = [s for s in anno if s["task"] in task_set]
        print(f"Filtered tasks: {task_set}")
    total_count = len(anno)
    anno = span_chunk(anno, chunk, index)
    print(f"Processing {len(anno)} samples (of {total_count} total)")

    done_keys = db_load_keys(db_path, lock_path)
    print(f"Already completed in DB: {len(done_keys)}")

    model = model_cls.from_pretrained_with_processor(
        bench_args.model_path, device=device
    )
    model.trigger_args = trigger_args

    prev_video = None
    cached_frames = None
    cached_timestamps = None
    local_cache = {}
    done_count = 0

    pbar = tqdm(
        total=len(anno),
        desc="Trigger detection",
        disable=(index != 0),
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for sample in anno:
        sample_key = make_sample_key(sample)

        if sample_key in done_keys:
            pbar.update(1)
            continue

        video_path = os.path.join(bench_args.data_path, sample["video"])
        try:
            if sample["video"] != prev_video:
                cached_frames, cached_timestamps = load_video_frames(
                    video_path, fps=1.0
                )
                prev_video = sample["video"]

            is_multi = {
                "default": sample["task"] in MULTI_EVENT_TASKS,
                "multi": True,
                "single": False,
            }[bench_args.event_mode]
            model.start_stream(query=sample["q"], is_multi_event=is_multi)
            segments = model.detect(cached_frames, cached_timestamps)
            local_cache[sample_key] = {"trigger_segments": segments}
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {video_path}: {e}")
            local_cache[sample_key] = {
                "trigger_segments": [
                    {"start_sec": 0.0, "end_sec": sample.get("duration", None)}
                ],
                "error": str(e),
            }
            prev_video = None

        if bench_args.verbose:
            print(f"\n[{sample['task']}] {sample['video']}")
            print(f"  Query: {sample['q'][:80]}...")
            print(f"  GT: {sample['tgt']}")
            print(f"  Result: {local_cache[sample_key]}")

        done_count += 1
        pbar.update(1)

        if len(local_cache) >= bench_args.flush_every:
            db_flush(db_path, lock_path, local_cache)
            pbar.write(f"  [flush] +{len(local_cache)} to DB")
            done_keys.update(local_cache.keys())
            local_cache.clear()

    if local_cache:
        db_flush(db_path, lock_path, local_cache)
        done_keys.update(local_cache.keys())
        local_cache.clear()

    pbar.close()
    db_maybe_export_jsonl(db_path, lock_path, total_count)
    print(f"[rank {index}] Done. Wrote {done_count} new samples to {db_path}")
