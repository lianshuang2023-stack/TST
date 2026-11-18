#!/usr/bin/env python3
"""Convert the user-provided 2024 CSV into the experiment's clean JSONL schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_splits import clean_article_text


LABEL_MAP = {"左倾": "left", "中性": "center", "右倾": "right"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "test.jsonl"
    counts = Counter()
    publishers = set()
    dates = []
    rows_written = 0
    rejected_empty_after_cleaning = 0
    with args.input.open(encoding="utf-8-sig", newline="") as source, output.open("w", encoding="utf-8") as target:
        for row in csv.DictReader(source):
            label = LABEL_MAP.get(row.get("article_lean_class", "").strip())
            if label is None:
                raise ValueError(f"Unexpected article_lean_class: {row.get('article_lean_class')!r}")
            publisher = row.get("publisher_full", "").strip()
            body = clean_article_text(row.get("article_body", ""), publisher=publisher)
            if not body:
                rejected_empty_after_cleaning += 1
                continue
            record = {
                "id": row.get("article_id", "").strip(),
                "publisher": publisher,
                "story_id": row.get("article_id", "").strip(),
                "date": row.get("datetime", "")[:10],
                "topic": row.get("topic", "").strip(),
                "headline": "",
                "stance_original": row.get("article_lean", "").strip(),
                "label": label,
                "text": body,
                "split": "external_test_2024",
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[label] += 1
            publishers.add(publisher)
            dates.append(record["date"])
            rows_written += 1

    manifest = {
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "articles": rows_written,
        "rejected_empty_after_cleaning": rejected_empty_after_cleaning,
        "labels": dict(counts),
        "publishers": len(publishers),
        "date_min": min(dates),
        "date_max": max(dates),
        "input_policy": "cleaned article_body only; CSV has no headline column; excludes URL, publisher, takeaways, annotation reasons, tone and sentence-level metadata",
        "label_policy": "article_lean_class: 左倾->left, 中性->center, 右倾->right",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
