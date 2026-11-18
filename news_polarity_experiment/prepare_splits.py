#!/usr/bin/env python3
"""Build leakage-resistant train/validation/test splits for AllSides articles.

Records are kept together when they share either a story_id or a normalized
article body.  The latter closes a gap in the supplied split: a small number
of identical article bodies occur in more than one split under different
story IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
LABELS = ("left", "center", "right")

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
BYLINE_RE = re.compile(
    r"^(?:by|written by|reported by|from the desk of|author)\s*[:\-–—]?\s+.+$",
    re.IGNORECASE,
)
FOOTER_RE = re.compile(
    r"^(?:copyright(?:\s+\d{4})?|©|all rights reserved|subscribe|sign up|follow us|read more|advertisement|advertising|terms of use|privacy policy)\b.*$",
    re.IGNORECASE,
)
WIRE_LEDE_RE = re.compile(
    r"^[A-Za-z][A-Za-z .,'’‘-]{1,80}\s*\((?:AP|Reuters|AFP|CNN|BBC|NPR)\)\s*[—–-]\s*",
)
REPORTER_CREDIT_RE = re.compile(
    r"(?:\s+|^)(?:reported|reports|wrote|writing|contributed|contributed to this report)\s+from\s+.+$",
    re.IGNORECASE,
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def clean_article_text(value: str, publisher: str = "", author: str = "") -> str:
    """Remove input-leaking metadata while preserving article wording.

    This is intentionally conservative: only URLs, standalone byline lines,
    exact publisher/author lines, and common footer boilerplate are removed.
    The raw article is never passed to a model.
    """
    publisher_norm = normalized_text(publisher)
    author_norm = normalized_text(author)
    cleaned_lines: list[str] = []
    for raw_line in str(value or "").splitlines():
        line = URL_RE.sub(" ", raw_line).strip()
        if not line:
            continue
        line_norm = normalized_text(line)
        if line_norm and line_norm in {publisher_norm, author_norm}:
            continue
        if BYLINE_RE.match(line):
            continue
        if FOOTER_RE.match(line):
            continue
        line = WIRE_LEDE_RE.sub("", line).strip()
        line = REPORTER_CREDIT_RE.sub("", line).strip()
        if not line:
            continue
        cleaned_lines.append(line)
    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def normalized_body(value: str) -> str:
    """Normalise harmless formatting differences before hashing."""
    value = normalized_text(value)
    return re.sub(r"\s+", " ", value).strip()


def body_hash(record: dict[str, Any]) -> str | None:
    body = normalized_body(str(record.get("text", "")))
    if not body:
        return None
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_split in SPLITS:
        path = input_dir / f"{source_split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["text"] = clean_article_text(
                    record.get("text", ""),
                    record.get("publisher", record.get("source", "")),
                    record.get("author", record.get("byline", "")),
                )
                # The model input policy is explicit. Keep only non-sensitive
                # metadata needed for grouping/auditing in the prepared JSONL.
                for key in ("url", "author", "byline"):
                    record.pop(key, None)
                record["_source_split"] = source_split
                record["_source_line"] = line_number
                records.append(record)
    return records


def make_components(records: list[dict[str, Any]]) -> list[list[int]]:
    union_find = UnionFind(len(records))
    first_story: dict[str, int] = {}
    first_body: dict[str, int] = {}

    for index, record in enumerate(records):
        story_id = str(record.get("story_id", "")).strip()
        if story_id:
            if story_id in first_story:
                union_find.union(index, first_story[story_id])
            else:
                first_story[story_id] = index

        fingerprint = body_hash(record)
        if fingerprint:
            if fingerprint in first_body:
                union_find.union(index, first_body[fingerprint])
            else:
                first_body[fingerprint] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union_find.find(index)].append(index)
    return list(components.values())


def component_stats(component: Iterable[int], records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for index in component:
        label = records[index].get("label")
        if label not in LABELS:
            raise ValueError(f"Unexpected label {label!r} in record {records[index].get('id')!r}")
        counts[str(label)] += 1
    return counts


def split_components(
    components: list[list[int]], records: list[dict[str, Any]], seed: int, trials: int
) -> dict[int, str]:
    """Assign connected components using article-level class-ratio targets.

    Several deterministic, seeded greedy assignments are considered and the
    assignment with the smallest normalized class-count error is retained.
    """
    stats = [component_stats(component, records) for component in components]
    totals = Counter()
    for count in stats:
        totals.update(count)
    targets = {
        split: {label: totals[label] * RATIOS[split] for label in LABELS}
        for split in SPLITS
    }

    def score(counts: dict[str, Counter[str]]) -> float:
        return sum(
            ((counts[split][label] - targets[split][label]) / max(1.0, targets[split][label])) ** 2
            for split in SPLITS
            for label in LABELS
        )

    best_assignment: dict[int, str] | None = None
    best_score = float("inf")
    for trial in range(trials):
        rng = random.Random(seed + trial)
        order = list(range(len(components)))
        # Larger groups are harder to place, while the random tie-break makes
        # multiple deterministic restarts useful.
        order.sort(key=lambda index: (-len(components[index]), rng.random()))
        counts = {split: Counter() for split in SPLITS}
        assignment: dict[int, str] = {}

        for component_index in order:
            component_count = stats[component_index]
            candidate_scores: list[tuple[float, str]] = []
            for split in SPLITS:
                counts[split].update(component_count)
                candidate_scores.append((score(counts), split))
                counts[split].subtract(component_count)
            min_score = min(value for value, _ in candidate_scores)
            best_splits = [split for value, split in candidate_scores if abs(value - min_score) < 1e-12]
            selected = rng.choice(best_splits)
            counts[selected].update(component_count)
            assignment[component_index] = selected

        assignment_score = score(counts)
        if assignment_score < best_score:
            best_assignment, best_score = assignment, assignment_score

    assert best_assignment is not None
    result: dict[int, str] = {}
    for component_index, component in enumerate(components):
        for record_index in component:
            result[record_index] = best_assignment[component_index]
    return result


def overlap_count(left: set[str], right: set[str]) -> int:
    return len(left.intersection(right))


def audit(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    story_sets = {
        split: {str(record.get("story_id")) for record in records if record.get("story_id")}
        for split, records in records_by_split.items()
    }
    body_sets = {
        split: {fingerprint for record in records if (fingerprint := body_hash(record))}
        for split, records in records_by_split.items()
    }
    return {
        "story_id_overlap": {
            "train_validation": overlap_count(story_sets["train"], story_sets["validation"]),
            "train_test": overlap_count(story_sets["train"], story_sets["test"]),
            "validation_test": overlap_count(story_sets["validation"], story_sets["test"]),
        },
        "normalized_text_overlap": {
            "train_validation": overlap_count(body_sets["train"], body_sets["validation"]),
            "train_test": overlap_count(body_sets["train"], body_sets["test"]),
            "validation_test": overlap_count(body_sets["validation"], body_sets["test"]),
        },
    }


def clean_for_output(record: dict[str, Any], split: str) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if not key.startswith("_")}
    result["split"] = split
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input_dir)
    components = make_components(records)
    assignment = split_components(components, records, args.seed, args.trials)
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for index, record in enumerate(records):
        split = assignment[index]
        output[split].append(clean_for_output(record, split))

    for split in SPLITS:
        output[split].sort(key=lambda record: str(record.get("id", "")))
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in output[split]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(args.input_dir.resolve()),
        "method": "group-stratified split over connected components of story_id and normalized text SHA-256",
        "seed": args.seed,
        "ratios": RATIOS,
        "total_articles": len(records),
        "connected_groups": len(components),
        "splits": {
            split: {
                "articles": len(output[split]),
                "stories": len({record["story_id"] for record in output[split]}),
                "labels": dict(Counter(record["label"] for record in output[split])),
            }
            for split in SPLITS
        },
        "leakage_audit": audit(output),
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
