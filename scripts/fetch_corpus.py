"""Fetch arXiv abstracts and write them to data/corpus.jsonl.

Uses the public arXiv Atom API (no auth). The API asks for a ~3s delay between
requests, so a 6k-document pull takes roughly two minutes.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "http://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

CATEGORIES = ["cs.IR", "cs.CL", "cs.LG", "cs.DB", "cs.SE", "cs.DC"]

PAGE_SIZE = 200
REQUEST_DELAY_S = 3.0


def fetch_page(category: str, start: int, page_size: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    req = urllib.request.Request(
        f"{API}?{params}", headers={"User-Agent": "hybrid-search-eval/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    docs = []
    for entry in root.findall("atom:entry", ATOM):
        arxiv_url = entry.findtext("atom:id", default="", namespaces=ATOM)
        arxiv_id = arxiv_url.rsplit("/", 1)[-1]
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ATOM).split())
        abstract = " ".join(
            entry.findtext("atom:summary", default="", namespaces=ATOM).split()
        )
        if not arxiv_id or not title or not abstract:
            continue

        authors = [
            a.findtext("atom:name", default="", namespaces=ATOM)
            for a in entry.findall("atom:author", ATOM)
        ]
        cats = [
            c.attrib.get("term", "") for c in entry.findall(f"{ARXIV_NS}category")
        ]
        docs.append(
            {
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": [a for a in authors if a][:10],
                "categories": [c for c in cats if c],
                "published": entry.findtext("atom:published", default="", namespaces=ATOM),
                "url": arxiv_url,
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    with args.out.open("w") as f:
        for category in CATEGORIES:
            fetched = 0
            while fetched < args.per_category:
                page_size = min(PAGE_SIZE, args.per_category - fetched)
                try:
                    docs = fetch_page(category, fetched, page_size)
                except Exception as exc:  # the arXiv API is occasionally flaky
                    print(f"  {category} start={fetched} failed ({exc}), retrying once")
                    time.sleep(REQUEST_DELAY_S * 2)
                    docs = fetch_page(category, fetched, page_size)

                if not docs:
                    break

                for doc in docs:
                    if doc["id"] in seen:
                        continue
                    seen.add(doc["id"])
                    f.write(json.dumps(doc) + "\n")
                    written += 1

                fetched += len(docs)
                print(f"  {category}: {fetched}/{args.per_category} (kept {written})")
                time.sleep(REQUEST_DELAY_S)

    print(f"\nWrote {written} unique documents to {args.out}")


if __name__ == "__main__":
    main()
