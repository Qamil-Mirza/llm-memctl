"""Write the generations a cache-only pass found missing (EXPERIMENTS.md §20), many at once.

A run with `cache_only: true` records each miss under <cache_dir>/_misses. This writes those generations through
the same cached model wrapper, so the cache keys are exactly the ones the run will ask for, then removes each
miss it filled. It is safe to stop and rerun: a filled miss is gone, and an entry already in the cache is not
generated again.

    python -m memctl.fill_cache --cache-dir cache/notes_qwen7b_exp20 --base-url URL/v1 --workers 32
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memctl.llm import build_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--backend", default="openai", help="stub for a dry run")
    parser.add_argument("--name", default="qwen2.5-7b-instruct")
    parser.add_argument("--base-url")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    llm = build_llm({"backend": args.backend, "name": args.name, "base_url": args.base_url, "timeout_s": 300,
                     "cache_dir": args.cache_dir})
    misses = sorted((Path(args.cache_dir) / "_misses").glob("*.json"))
    started = time.time()

    def fill(path: Path) -> int:
        request = json.loads(path.read_text())
        output = llm.generate(request["prompt"], max_new_tokens=request["max_new_tokens"])
        path.unlink()
        return len(output)

    with ThreadPoolExecutor(args.workers) as pool:
        done = list(pool.map(fill, misses))
    print(json.dumps({"filled": len(done), "already_cached": llm.cache_hits, "seconds": round(time.time() - started, 1),
                      "left": len(list((Path(args.cache_dir) / "_misses").glob("*.json")))}))


if __name__ == "__main__":
    main()
