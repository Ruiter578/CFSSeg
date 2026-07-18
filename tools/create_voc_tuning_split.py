#!/usr/bin/env python3
"""Create a deterministic class-aware tuning split from VOC train_cls.txt."""

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitResult:
    holdout_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    seed: int
    fraction: float
    source_count: int
    source_class_counts: dict[int, int]
    holdout_class_counts: dict[int, int]


def parse_class_rows(lines: Iterable[str]) -> list[tuple[str, tuple[int, ...]]]:
    rows = []
    seen = set()
    for line_number, line in enumerate(lines, start=1):
        tokens = line.split()
        if not tokens:
            continue
        image_id, *class_tokens = tokens
        if image_id in seen:
            raise ValueError(f"Duplicate image id at line {line_number}: {image_id}")
        try:
            classes = tuple(sorted({int(token) for token in class_tokens}))
        except ValueError as exc:
            raise ValueError(f"Invalid class token at line {line_number}: {line!r}") from exc
        seen.add(image_id)
        rows.append((image_id, classes))
    if not rows:
        raise ValueError("No samples were found in the source class list.")
    return rows


def _rank(seed: int, image_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def _class_counts(rows: Iterable[tuple[str, tuple[int, ...]]]) -> Counter:
    return Counter(cls for _, classes in rows for cls in classes)


def build_split(rows, fraction: float, seed: int) -> SplitResult:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")

    canonical_rows = sorted(rows, key=lambda row: row[0])
    if len({image_id for image_id, _ in canonical_rows}) != len(canonical_rows):
        raise ValueError("Image ids must be unique.")

    holdout_size = math.floor(len(canonical_rows) * fraction + 0.5)
    if holdout_size <= 0 or holdout_size >= len(canonical_rows):
        raise ValueError("fraction produces an empty train or holdout split")

    source_counts = _class_counts(canonical_rows)
    target_counts = {
        cls: max(1, int(round(count * fraction)))
        for cls, count in source_counts.items()
    }
    ranked_rows = sorted(canonical_rows, key=lambda row: (_rank(seed, row[0]), row[0]))
    remaining = {image_id: classes for image_id, classes in ranked_rows}
    class_by_id = dict(canonical_rows)
    selected = []
    selected_counts = Counter()

    if holdout_size >= len(source_counts):
        for cls in sorted(source_counts):
            if selected_counts[cls]:
                continue
            image_id = next(
                image_id for image_id, classes in remaining.items() if cls in classes
            )
            classes = remaining.pop(image_id)
            selected.append(image_id)
            selected_counts.update(classes)

    while len(selected) < holdout_size:
        best_id = None
        best_score = None
        for image_id, classes in remaining.items():
            score = sum(
                max(0.0, (target_counts[cls] - selected_counts[cls]) / target_counts[cls])
                for cls in classes
            )
            candidate = (score, -_rank(seed, image_id), image_id)
            if best_score is None or candidate > best_score:
                best_id = image_id
                best_score = candidate

        classes = remaining.pop(best_id)
        selected.append(best_id)
        selected_counts.update(classes)

    while True:
        current_coverage = sum(count > 0 for count in selected_counts.values())
        best_repair = None
        for index, selected_id in enumerate(selected):
            without_selected = selected_counts.copy()
            without_selected.subtract(class_by_id[selected_id])
            for candidate_id, candidate_classes in remaining.items():
                candidate_counts = without_selected.copy()
                candidate_counts.update(candidate_classes)
                candidate_coverage = sum(count > 0 for count in candidate_counts.values())
                candidate_key = (candidate_coverage, -_rank(seed, candidate_id), candidate_id)
                if candidate_coverage > current_coverage and (
                    best_repair is None or candidate_key > best_repair[0]
                ):
                    best_repair = (candidate_key, index, selected_id, candidate_id)
        if best_repair is None:
            break
        _, index, selected_id, candidate_id = best_repair
        selected_counts.subtract(class_by_id[selected_id])
        selected_counts.update(class_by_id[candidate_id])
        selected[index] = candidate_id
        remaining[selected_id] = class_by_id[selected_id]
        remaining.pop(candidate_id)

    holdout_ids = tuple(sorted(selected))
    holdout_set = set(holdout_ids)
    train_ids = tuple(image_id for image_id, _ in canonical_rows if image_id not in holdout_set)
    return SplitResult(
        holdout_ids=holdout_ids,
        train_ids=train_ids,
        seed=seed,
        fraction=fraction,
        source_count=len(canonical_rows),
        source_class_counts=dict(sorted(source_counts.items())),
        holdout_class_counts=dict(sorted(selected_counts.items())),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(split: SplitResult, output_path: Path, metadata_path: Path, source_path: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(split.holdout_ids) + "\n", encoding="utf-8")
    metadata = {
        "protocol": "voc_train_cls_holdout_v1",
        "source_path": source_path,
        "source_sha256": sha256_file(Path(source_path)) if Path(source_path).is_file() else None,
        "source_count": split.source_count,
        "seed": split.seed,
        "fraction": split.fraction,
        "holdout_count": len(split.holdout_ids),
        "train_count": len(split.train_ids),
        "source_class_counts": split.source_class_counts,
        "holdout_class_counts": split.holdout_class_counts,
        "holdout_sha256": sha256_file(output_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    rows = parse_class_rows(args.source.read_text(encoding="utf-8").splitlines())
    split = build_split(rows, fraction=args.fraction, seed=args.seed)
    write_split(split, args.holdout_output, args.metadata_output, str(args.source))
    print(
        f"Created holdout: {args.holdout_output} "
        f"({len(split.holdout_ids)}/{split.source_count}, seed={split.seed})"
    )


if __name__ == "__main__":
    main()
