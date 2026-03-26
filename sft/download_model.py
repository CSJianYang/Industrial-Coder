"""
Download a model or dataset from Hugging Face Hub.

Usage:
    # Download a model
    python download_model.py \
        --repo_id Multilingual-Multimodal-NLP/IndustrialCoder-Base \
        --local_dir /path/to/save/model

    # Download a dataset
    python download_model.py \
        --repo_id username/dataset-name \
        --repo_type dataset \
        --local_dir /path/to/save/dataset \
        --workers 32

    # Set HF token via environment variable (recommended):
    export HF_TOKEN=your_token_here
    python download_model.py --repo_id ...

    # Or pass token directly (not recommended for shared environments):
    python download_model.py --repo_id ... --token your_token_here
"""

import argparse
import os
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download a model or dataset from Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID, e.g. Multilingual-Multimodal-NLP/IndustrialCoder-Base")
    parser.add_argument("--local_dir", type=str, required=True,
                        help="Local directory to save the downloaded files")
    parser.add_argument("--repo_type", type=str, default="model", choices=["model", "dataset", "space"],
                        help="Type of the repository (default: model)")
    parser.add_argument("--revision", type=str, default=None,
                        help="Specific revision (branch, tag, or commit hash) to download")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache",
                        help="Cache directory for snapshot_download (default: ./hf_cache)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent download workers (default: 8)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token. Defaults to HF_TOKEN env variable if not set.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve token: explicit arg > HF_TOKEN env var > None (public repos only)
    token = args.token or os.environ.get("HF_TOKEN", None)

    print(f"=== HuggingFace Download ===")
    print(f"Repo:       {args.repo_id}  ({args.repo_type})")
    print(f"Revision:   {args.revision or 'latest'}")
    print(f"Local dir:  {args.local_dir}")
    print(f"Workers:    {args.workers}")
    print(f"Token:      {'set' if token else 'not set (public repo only)'}")
    print(f"===========================")

    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        max_workers=args.workers,
        resume_download=True,
        token=token,
    )

    print(f"Downloaded to: {downloaded_path}")


if __name__ == "__main__":
    main()
