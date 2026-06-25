#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import argparse
import os
import re
import tarfile
from collections import defaultdict

from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = "interlive/stream-data"
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".webm", ".m4v", ".mov", ".mpg", ".3gp"}


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract videos from the interlive/stream-data tars"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output video_root directory",
    )
    parser.add_argument(
        "--max_per_folder",
        type=int,
        default=0,
        help="Max videos to extract per internal folder per source (0 = extract all)",
    )
    args = parser.parse_args()

    # List all tar files under videos/ in the HF repo
    print(f"Listing files in {REPO_ID}/videos/ ...")
    tar_names = sorted(
        entry.rfilename
        for entry in list_repo_tree(REPO_ID, path_in_repo="videos", repo_type="dataset")
        if entry.rfilename.endswith(".tar")
    )
    print(f"Found {len(tar_names)} tar files")

    # Group by source prefix (everything before _partXX.tar)
    source_tars = defaultdict(list)
    for name in tar_names:
        basename = os.path.basename(name)
        match = re.match(r"(.+)_part\d+\.tar$", basename)
        if match:
            source_tars[match.group(1)].append(name)

    print(f"Sources: {len(source_tars)}")
    print(f"Output: {args.output_dir}")
    print(
        f"Max per folder: {args.max_per_folder if args.max_per_folder > 0 else 'all'}"
    )

    for source, tars in sorted(source_tars.items()):
        print(f"\n{'=' * 60}")
        print(f"{source} ({len(tars)} parts)")
        print(f"{'=' * 60}")

        total = 0
        target = os.path.join(args.output_dir, source)

        for tar_name in tars:
            part_label = os.path.basename(tar_name)
            print(f"  Downloading {part_label} ...", end=" ", flush=True)

            tar_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=tar_name,
                repo_type="dataset",
            )

            folder_counts = defaultdict(int)
            part_extracted = 0
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    if os.path.splitext(member.name)[1].lower() not in VIDEO_EXT:
                        continue

                    folder = os.path.dirname(member.name) or "."
                    if (
                        args.max_per_folder > 0
                        and folder_counts[folder] >= args.max_per_folder
                    ):
                        continue

                    dest = os.path.realpath(os.path.join(target, member.name))
                    if not dest.startswith(os.path.realpath(target)):
                        continue

                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with tf.extractfile(member) as src, open(dest, "wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)

                    folder_counts[folder] += 1
                    total += 1
                    part_extracted += 1

            print(f"extracted {part_extracted}")

        print(f"  Total: {total}")


if __name__ == "__main__":
    main()
