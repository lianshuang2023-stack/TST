#!/usr/bin/env python3
"""Prepare a sealed test set from unused AllSides stories and articles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from prepare_splits import clean_article_text, normalized_body, normalized_text


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def story_slug(url: str) -> str:
    return urlparse(str(url)).path.rstrip("/").split("/")[-1]


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--used-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-words", type=int, default=100)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    used = []
    for split in ("train", "validation", "test"):
        used.extend(load_jsonl(args.used_data_dir / f"{split}.jsonl"))
    used_story = {str(row.get("story_id", "")) for row in used}
    used_headline = {normalized_text(row.get("headline", "")) for row in used}
    used_body = {sha(normalized_body(row.get("text", ""))) for row in used if row.get("text")}

    selected = []
    seen_url: set[str] = set()
    seen_body: set[str] = set()
    rejected = Counter()
    for story in load_jsonl(args.combined):
        slug = story_slug(story.get("headline_link", ""))
        if not slug or slug in used_story:
            rejected["story_seen_in_original_splits"] += 1
            continue
        for label in ("left", "center", "right"):
            articles = []
            if isinstance(story.get(label), dict):
                articles.append(story[label])
            articles.extend(story.get(f"more_{label}", []) or [])
            for article in articles:
                url = str(article.get("link") or article.get("allsides_link") or "").strip()
                headline = str(article.get("headline") or "").strip()
                raw_text = str(article.get("content") or article.get("summary") or "").strip()
                publisher = str(article.get("source") or "").strip()
                if not url or url in seen_url:
                    rejected["missing_or_duplicate_url"] += 1
                    continue
                cleaned = clean_article_text(raw_text, publisher=publisher)
                body_fingerprint = sha(normalized_body(cleaned))
                if body_fingerprint in used_body or body_fingerprint in seen_body:
                    rejected["body_overlap"] += 1
                    continue
                if normalized_text(headline) in used_headline:
                    rejected["headline_overlap"] += 1
                    continue
                if len(cleaned.split()) < args.min_words:
                    rejected["too_short"] += 1
                    continue
                if raw_text.rstrip().endswith("...") or raw_text.rstrip().endswith("…"):
                    rejected["explicitly_truncated"] += 1
                    continue
                record_id = f"independent::{sha(url)[:16]}::{label}"
                selected.append(
                    {
                        "id": record_id,
                        "publisher": publisher,
                        "story_id": slug,
                        "date": story.get("date", ""),
                        "topic": story.get("topic", ""),
                        "headline": headline,
                        "stance_original": article.get("rating", label),
                        "label": label,
                        "text": cleaned,
                        "split": "independent_test",
                        "source_url_sha256": sha(url),
                    }
                )
                seen_url.add(url)
                seen_body.add(body_fingerprint)

    selected.sort(key=lambda row: row["id"])
    output = args.output_dir / "test.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_hash = sha(output.read_text(encoding="utf-8"))
    manifest = {
        "source": str(args.combined.resolve()),
        "used_data_dir": str(args.used_data_dir.resolve()),
        "selection_created_before_model_evaluation": True,
        "selection_policy": [
            "AllSides story_id absent from all original train/validation/test splits",
            "headline and normalized body absent from all original splits",
            f"cleaned body has at least {args.min_words} whitespace-delimited words",
            "source text does not end with an explicit ellipsis truncation marker",
            "labels collapsed to left/center/right from the AllSides side bucket",
        ],
        "articles": len(selected),
        "stories": len({row["story_id"] for row in selected}),
        "publishers": len({row["publisher"] for row in selected}),
        "labels": dict(Counter(row["label"] for row in selected)),
        "date_min": min(row["date"] for row in selected),
        "date_max": max(row["date"] for row in selected),
        "rejected": dict(rejected),
        "test_jsonl_sha256": output_hash,
        "warning": "Do not inspect model results on this set until the final candidate is frozen.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
