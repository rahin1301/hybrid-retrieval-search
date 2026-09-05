"""Print the judgement pool compactly for manual labelling.

Usage:
    python scripts/show_pool.py                # all queries
    python scripts/show_pool.py --qids q01 q02 # selected queries
    python scripts/show_pool.py --unjudged     # only candidates missing a grade
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qids", nargs="*", help="restrict to these query ids")
    parser.add_argument(
        "--unjudged", action="store_true", help="hide candidates that already have a grade"
    )
    parser.add_argument("--title-only", action="store_true", default=True)
    args = parser.parse_args()

    settings = get_settings()
    pool_path = settings.eval_dir / "pool.jsonl"
    qrels_path = settings.eval_dir / "qrels.json"

    if not pool_path.exists():
        raise SystemExit(f"No pool at {pool_path}. Run scripts/make_pool.py first.")

    qrels = json.loads(qrels_path.read_text()) if qrels_path.exists() else {}

    with pool_path.open() as handle:
        for line in handle:
            entry = json.loads(line)
            if args.qids and entry["qid"] not in args.qids:
                continue

            judged = qrels.get(entry["qid"], {})
            print(f"\n{'=' * 100}")
            print(f"{entry['qid']}: {entry['query']}")
            print("=" * 100)

            for candidate in entry["candidates"]:
                doc_id = candidate["doc_id"]
                grade = judged.get(doc_id)
                if args.unjudged and grade is not None:
                    continue
                marker = "·".join(candidate["found_by"])
                grade_str = f"[{grade}]" if grade is not None else "[ ]"
                print(f"{grade_str} {doc_id:16s} ({marker:11s}) {candidate['title']}")


if __name__ == "__main__":
    main()
