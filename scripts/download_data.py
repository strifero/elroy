"""Chapter 1, step 1: download the raw data.

    python scripts/download_data.py            # everything the Elroy-350M run uses
    python scripts/download_data.py --mini     # a small slice for the mini config

Four sources, all from the Hugging Face Hub:

  stack-edu        HuggingFaceTB/stack-edu, Python split. An INDEX only: blob ids
                   plus quality scores. scripts/fetch_stack_edu.py turns it into text.
  starcoderdata    bigcode/starcoderdata, python split. Gated: accept the terms on
                   the dataset page once, and have `hf auth login` done (or HF_TOKEN set).
  fineweb-edu      HuggingFaceFW/fineweb-edu, the 10BT sample.
  cosmopedia       HuggingFaceTB/smollm-corpus, cosmopedia-v2 split, first N shards.

Everything lands under data/raw/<source>/ exactly as the Hub stores it, so the
tokenizer step can read parquet with pyarrow and nothing else. The Hub downloader
is resumable; rerunning this script only fetches what is missing.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

RAW = os.path.join("data", "raw")

# (repo_id, local subdir, include patterns for the full run, include patterns for --mini)
SOURCES = [
    ("HuggingFaceTB/stack-edu", "stack-edu",
     ["Python/*"], ["Python/train-00000-of-00005.parquet"]),
    ("bigcode/starcoderdata", "starcoderdata",
     ["python/train-000[0-1]*"], ["python/train-00000-of-00059.parquet"]),
    ("HuggingFaceFW/fineweb-edu", "fineweb-edu",
     ["sample/10BT/*"], ["sample/10BT/000_00000.parquet"]),
    ("HuggingFaceTB/smollm-corpus", "smollm-corpus",
     ["cosmopedia-v2/train-0000[0-9]-*"], ["cosmopedia-v2/train-00000-*"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true", help="one shard per source")
    ap.add_argument("--only", nargs="*", help="subset of source names to fetch")
    args = ap.parse_args()

    for repo_id, subdir, full, mini in SOURCES:
        if args.only and subdir not in args.only:
            continue
        patterns = mini if args.mini else full
        dest = os.path.join(RAW, subdir)
        print(f"==> {repo_id} -> {dest}  patterns={patterns}", flush=True)
        try:
            snapshot_download(repo_id, repo_type="dataset", allow_patterns=patterns,
                              local_dir=dest, max_workers=8)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "gated" in msg.lower() or "403" in msg or "requires approval" in msg.lower():
                print(f"    {repo_id} is gated. Accept the terms at "
                      f"https://huggingface.co/datasets/{repo_id} while logged in, "
                      f"make sure `hf auth login` has been run, then rerun.", file=sys.stderr)
                sys.exit(1)
            raise
    print("done")


if __name__ == "__main__":
    main()
