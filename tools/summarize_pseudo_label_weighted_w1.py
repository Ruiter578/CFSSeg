#!/usr/bin/env python
"""Validate and summarize the pre-registered weighted pseudo-label W1 screen."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_pseudo_label_grid import group_miou, read_grid


EXPECTED_PAIRS = {
    ("overlap", "confidence"),
    ("overlap", "confidence_margin"),
    ("disjoint", "confidence"),
    ("disjoint", "confidence_margin"),
}
EXPECTED_THRESHOLDS = {
    "overlap": 0.447265625,
    "disjoint": 0.029296875,
}
EXPECTED_BASE_SUBPATHS = {
    "overlap": "20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32",
    "disjoint": "20260705_pseudo_15-5_disjoint_off_seed1_bs32",
}
W1_MIN_MEAN_ALL_DELTA_PP = 0.10
W1_MIN_PER_SETTING_ALL_DELTA_PP = -0.05
W1_MIN_NEW_DELTA_PP = -0.10
W1_MAX_EXTREME_MASS_RATIO = 0.95


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_and_validate_grid(path):
    rows = read_grid(path)
    if len(rows) != 4:
        raise ValueError(f"W1 grid must contain exactly four rows, got {len(rows)}.")
    pairs = {(row["setting"], row.get("weighting", "")) for row in rows}
    if pairs != EXPECTED_PAIRS:
        raise ValueError(f"W1 setting/weighting pairs are not pre-registered: {pairs}")
    if len({row["subpath"] for row in rows}) != len(rows):
        raise ValueError("W1 subpaths must be unique.")
    for row in rows:
        setting = row["setting"]
        if row["strategy"] != "fixed":
            raise ValueError("W1 must use fixed matched-global acceptance.")
        if abs(float(row["confidence"]) - EXPECTED_THRESHOLDS[setting]) > 1e-12:
            raise ValueError(f"W1 {setting} threshold does not match pre-registration.")
        if row["base_subpath"] != EXPECTED_BASE_SUBPATHS[setting]:
            raise ValueError(
                f"W1 {setting} base_subpath={row['base_subpath']!r}, "
                f"expected {EXPECTED_BASE_SUBPATHS[setting]!r}."
            )
        expected = {
            "task": "15-5",
            "skip_step0": "1",
            "batch_size": "32",
            "step0_batch_size": "32",
            "buffer": "8196",
            "gamma": "1",
            "random_seed": "1",
            "model": "deeplabv3_resnet101",
            "air_feature_source": "auto",
            "quantile": "0.01",
            "margin_min": "0.0",
            "min_conf": "0.0",
            "max_conf": "1.0",
            "min_pixels": "1",
            "shrinkage": "0.0",
            "threshold_artifact": "",
            "threshold_max_batches": "0",
        }
        for key, value in expected.items():
            if row[key] != value:
                raise ValueError(
                    f"W1 row {row['name']} has {key}={row[key]!r}, expected {value!r}."
                )
    return rows


def _resolve_path(path, repo_root):
    path = Path(path)
    return path if path.is_absolute() else Path(repo_root) / path


def load_and_verify_baselines(path, *, repo_root):
    registry_path = _resolve_path(path, repo_root)
    registry = read_json(registry_path)
    if registry is None:
        raise FileNotFoundError(f"Missing W1 baseline registry: {path}")
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported W1 baseline registry schema.")
    settings = registry.get("settings", {})
    if set(settings) != {"overlap", "disjoint"}:
        raise ValueError("W1 baseline registry must contain overlap and disjoint.")
    w0_audit = registry.get("w0_audit", {})
    if set(w0_audit) != {"overlap", "disjoint"}:
        raise ValueError("W1 baseline registry must bind both W0 audit summaries.")

    verified = {}
    registry_sha = file_sha256(registry_path)
    for setting, record in settings.items():
        result_path = _resolve_path(record["result_path"], repo_root)
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing W1 baseline result: {result_path}")
        actual_sha = file_sha256(result_path)
        if actual_sha != record["result_sha256"]:
            raise ValueError(
                f"Baseline result SHA256 mismatch for {setting}: "
                f"expected={record['result_sha256']} actual={actual_sha}"
            )
        result = read_json(result_path)
        if set((result.get("Class IoU") or {})) != {
            str(class_id) for class_id in range(21)
        }:
            raise ValueError(f"Baseline {setting} Class IoU keys mismatch.")
        old_miou, new_miou = group_miou(result)
        actual_metrics = {
            "all_miou": result.get("Mean IoU"),
            "old_miou": old_miou,
            "new_miou": new_miou,
        }
        for key, expected in record["metrics"].items():
            if abs(float(actual_metrics[key]) - float(expected)) > 1e-12:
                raise ValueError(
                    f"Baseline {setting} {key} mismatch: "
                    f"{actual_metrics[key]} vs {expected}"
                )
        stats = read_json(result_path.parent / "pseudo_label_stats.json")
        manifest = read_json(result_path.parent / "run_manifest.json")
        if stats is None or manifest is None:
            raise ValueError(f"Baseline {setting} is missing stats or manifest.")
        expected_manifest = {
            "task": "15-5",
            "setting": setting,
            "curr_step": 1,
            "base_subpath": EXPECTED_BASE_SUBPATHS[setting],
            "model": "deeplabv3_resnet101",
            "batch_size": 32,
            "buffer": 8196,
            "random_seed": 1,
            "pseudo_label_strategy": "fixed",
            "requested_air_feature_source": "auto",
            "resolved_air_feature_source": "decoder",
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Baseline {setting} manifest {key}={manifest.get(key)!r}, "
                    f"expected {expected!r}."
                )
        if abs(float(manifest.get("gamma", -1)) - 1.0) > 1e-12:
            raise ValueError(f"Baseline {setting} manifest gamma mismatch.")
        if (
            abs(
                float(manifest.get("pseudo_label_confidence", -1))
                - float(record["threshold"])
            )
            > 1e-12
        ):
            raise ValueError(f"Baseline {setting} manifest threshold mismatch.")
        if manifest.get("subpath") != result_path.parents[4].name:
            raise ValueError(f"Baseline {setting} manifest subpath mismatch.")
        if int(stats.get("candidate_count", -1)) != int(record["candidate_count"]):
            raise ValueError(f"Baseline {setting} candidate_count mismatch.")
        if int(stats.get("accepted_count", -1)) != int(record["accepted_count"]):
            raise ValueError(f"Baseline {setting} accepted_count mismatch.")
        if stats.get("teacher_sha256") != record["teacher_sha256"]:
            raise ValueError(f"Baseline {setting} teacher SHA mismatch.")
        if manifest.get("base_checkpoint_sha256") != record["teacher_sha256"]:
            raise ValueError(f"Baseline {setting} manifest teacher SHA mismatch.")
        audit_record = w0_audit[setting]
        audit_path = _resolve_path(audit_record["summary_path"], repo_root)
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing W0 reliability summary: {audit_path}")
        audit_sha = file_sha256(audit_path)
        if audit_sha != audit_record["summary_sha256"]:
            raise ValueError(
                f"W0 summary SHA256 mismatch for {setting}: "
                f"expected={audit_record['summary_sha256']} actual={audit_sha}"
            )
        audit = read_json(audit_path)
        directions = {}
        for weighting in ("confidence", "confidence_margin"):
            signal = (audit.get("signals") or {}).get(weighting) or {}
            failed = []
            if int(signal.get("nonempty_equal_count_bins", 0)) < 8:
                failed.append("nonempty_bins")
            if float(signal.get("spearman_rho", -1.0)) < 0.6:
                failed.append("spearman_rho")
            if float(signal.get("top_bottom_precision_gap", -1.0)) < 0.05:
                failed.append("precision_gap")
            if (
                float(signal.get("extreme_mass_ratio", 1.0))
                > W1_MAX_EXTREME_MASS_RATIO
            ):
                failed.append("extreme_mass_ratio")
            directions[weighting] = {
                "passed": not failed,
                "failed": failed,
                "summary_path": str(audit_path),
                "summary_sha256": audit_sha,
            }
        verified[setting] = {
            **record,
            "result_path_record": record["result_path"],
            "result_path": str(result_path),
            "registry_path_record": str(path),
            "registry_sha256": registry_sha,
            "metrics": actual_metrics,
            "class_iou": {
                str(class_id): float(value)
                for class_id, value in (result.get("Class IoU") or {}).items()
            },
            "per_class_candidates": stats.get("per_class_candidates") or {},
            "per_class_accepted": stats.get("per_class_accepted") or {},
            "w0_direction": directions,
        }
    return verified


def verify_step0_checkpoints(rows, *, repo_root, baselines):
    verified = {}
    for row in rows:
        setting = row["setting"]
        checkpoint = (
            Path(repo_root)
            / "checkpoints"
            / row["base_subpath"]
            / "voc"
            / row["task"]
            / setting
            / "step0"
            / f"{row['model']}_voc_{row['task']}_step_0_{setting}.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing W1 step0 checkpoint: {checkpoint}")
        actual_sha = file_sha256(checkpoint)
        expected_sha = baselines[setting]["teacher_sha256"]
        if actual_sha != expected_sha:
            raise ValueError(
                f"W1 step0 SHA256 mismatch for {setting}: "
                f"expected={expected_sha} actual={actual_sha}"
            )
        if setting in verified and verified[setting]["path"] != str(checkpoint):
            raise ValueError(f"W1 rows disagree on {setting} step0 checkpoint.")
        verified[setting] = {
            "path": str(checkpoint),
            "sha256": actual_sha,
        }
    if set(verified) != {"overlap", "disjoint"}:
        raise ValueError("W1 step0 verification must cover overlap and disjoint.")
    return verified


def verify_output_paths_absent(rows, *, repo_root):
    for row in rows:
        output_root = Path(repo_root) / "checkpoints" / row["subpath"]
        if output_root.exists():
            raise FileExistsError(
                f"W1 output root already exists: {output_root}"
            )


def _weight_extreme_mass_ratio(stats):
    histogram = stats.get("weight_histogram") or {}
    counts = histogram.get("counts") or []
    total = int(stats.get("weighted_pixel_count") or 0)
    if total <= 0 or len(counts) != 20:
        return None
    return max(int(counts[0]), int(counts[-1])) / total


def summarize_candidate_row(row, *, checkpoints_root, baselines):
    output_dir = (
        Path(checkpoints_root)
        / row["subpath"]
        / "voc"
        / row["task"]
        / row["setting"]
        / "step1"
    )
    result_paths = (
        sorted(output_dir.glob("test_results_*.json"))
        if output_dir.is_dir()
        else []
    )
    result_path = result_paths[0] if len(result_paths) == 1 else None
    summary = {
        "name": row["name"],
        "subpath": row["subpath"],
        "setting": row["setting"],
        "weighting": row["weighting"],
        "status": "missing_output" if not output_dir.exists() else "missing_result",
        "output_dir": str(output_dir),
    }
    if len(result_paths) > 1:
        summary["status"] = "ambiguous_result"
        return summary
    if result_path is None:
        return summary

    result = read_json(result_path)
    stats = read_json(output_dir / "pseudo_label_stats.json")
    manifest = read_json(output_dir / "run_manifest.json")
    if stats is None or manifest is None:
        summary["status"] = "missing_metadata"
        return summary

    old_miou, new_miou = group_miou(result)
    metrics = {
        "all_miou": float(result["Mean IoU"]),
        "old_miou": float(old_miou),
        "new_miou": float(new_miou),
    }
    baseline = baselines[row["setting"]]
    deltas = {
        f"delta_{name}": metrics[f"{name}_miou"] - baseline["metrics"][f"{name}_miou"]
        for name in ("all", "old", "new")
    }
    candidate_count = int(stats.get("candidate_count", -1))
    accepted_count = int(stats.get("accepted_count", -1))
    per_class_candidates = stats.get("per_class_candidates") or {}
    per_class_accepted = stats.get("per_class_accepted") or {}
    histogram_counts = (stats.get("weight_histogram") or {}).get("counts") or []
    histogram_ok = (
        stats.get("schema_version") == 2
        and len(histogram_counts) == 20
        and sum(int(count) for count in histogram_counts)
        == int(stats.get("weighted_pixel_count", -1))
    )
    weight_stat_names = (
        "weight_mean",
        "weight_std",
        "weight_p10",
        "weight_p25",
        "weight_p50",
        "weight_p75",
        "weight_p90",
        "weight_min",
        "weight_max",
    )
    weight_values = [stats.get(name) for name in weight_stat_names]
    weight_stats_ok = (
        all(
            value is not None and math.isfinite(float(value))
            for value in weight_values
        )
        and float(stats["weight_std"]) >= 0.0
        and all(
            0.0 <= float(stats[name]) <= 1.0
            for name in weight_stat_names
            if name != "weight_std"
        )
    )
    mask_counts_match = (
        candidate_count == int(baseline["candidate_count"])
        and accepted_count == int(baseline["accepted_count"])
        and int(stats.get("weighted_pixel_count", -1)) == accepted_count
        and per_class_candidates == baseline["per_class_candidates"]
        and per_class_accepted == baseline["per_class_accepted"]
    )
    candidate_class_iou = {
        str(class_id): float(value)
        for class_id, value in (result.get("Class IoU") or {}).items()
    }
    result_ok = (
        set(candidate_class_iou) == set(baseline["class_iou"])
        and len(candidate_class_iou) == 21
        and all(math.isfinite(value) for value in metrics.values())
        and all(math.isfinite(value) for value in candidate_class_iou.values())
    )
    per_class_iou_delta = {
        class_id: candidate_class_iou[class_id] - baseline_value
        for class_id, baseline_value in baseline["class_iou"].items()
        if class_id in candidate_class_iou
    }
    source_status_path = _resolve_path(
        manifest.get("source_status_path", ""),
        REPO_ROOT,
    )
    source_patch_path = _resolve_path(
        manifest.get("source_patch_path", ""),
        REPO_ROOT,
    )
    source_dirty = manifest.get("source_dirty")
    source_commit = manifest.get("source_commit")
    provenance_ok = (
        isinstance(source_dirty, bool)
        and isinstance(source_commit, str)
        and len(source_commit) == 40
        and source_status_path.is_file()
        and source_patch_path.is_file()
        and (
            source_dirty
            == bool(source_status_path.read_text(encoding="utf-8").strip())
        )
    )
    manifest_ok = (
        manifest.get("task") == row["task"]
        and manifest.get("setting") == row["setting"]
        and int(manifest.get("curr_step", -1)) == 1
        and manifest.get("subpath") == row["subpath"]
        and manifest.get("base_subpath") == row["base_subpath"]
        and manifest.get("model") == row["model"]
        and int(manifest.get("batch_size", -1)) == int(row["batch_size"])
        and int(manifest.get("buffer", -1)) == int(row["buffer"])
        and abs(float(manifest.get("gamma", -1)) - float(row["gamma"])) <= 1e-12
        and manifest.get("rhl_norm") == "none"
        and int(manifest.get("rhl_seed", 0)) == -1
        and manifest.get("pin_memory") == "0"
        and manifest.get("requested_air_feature_source")
        == row["air_feature_source"]
        and manifest.get("resolved_air_feature_source") == "decoder"
        and manifest.get("pseudo_label_strategy") == "fixed"
        and manifest.get("pseudo_label_weighting") == row["weighting"]
        and abs(
            float(manifest.get("pseudo_label_confidence", -1))
            - float(row["confidence"])
        )
        <= 1e-12
        and abs(
            float(manifest.get("pseudo_label_quantile", -1))
            - float(row["quantile"])
        )
        <= 1e-12
        and abs(
            float(manifest.get("pseudo_label_min_conf", -1))
            - float(row["min_conf"])
        )
        <= 1e-12
        and abs(
            float(manifest.get("pseudo_label_max_conf", -1))
            - float(row["max_conf"])
        )
        <= 1e-12
        and int(manifest.get("pseudo_label_min_pixels", -1))
        == int(row["min_pixels"])
        and abs(
            float(manifest.get("pseudo_label_shrinkage", -1))
            - float(row["shrinkage"])
        )
        <= 1e-12
        and abs(
            float(manifest.get("pseudo_label_margin_min", -1))
            - float(row["margin_min"])
        )
        <= 1e-12
        and manifest.get("pseudo_label_threshold_artifact") is None
        and int(manifest.get("pseudo_label_threshold_max_batches", -1))
        == int(row["threshold_max_batches"])
        and int(manifest.get("random_seed", -1)) == 1
        and manifest.get("base_checkpoint_sha256") == baseline["teacher_sha256"]
        and manifest.get("teacher_sha256") == baseline["teacher_sha256"]
        and provenance_ok
        and manifest.get("baseline_result_path") == baseline["result_path_record"]
        and manifest.get("baseline_result_sha256") == baseline["result_sha256"]
        and manifest.get("baseline_registry_path")
        == baseline["registry_path_record"]
        and manifest.get("baseline_registry_sha256")
        == baseline["registry_sha256"]
        and stats.get("weighting") == row["weighting"]
        and stats.get("task") == row["task"]
        and stats.get("setting") == row["setting"]
        and int(stats.get("step", -1)) == 1
        and stats.get("strategy") == "fixed"
        and abs(
            float(stats.get("fixed_confidence", -1))
            - float(row["confidence"])
        )
        <= 1e-12
        and stats.get("teacher_sha256") == baseline["teacher_sha256"]
        and histogram_ok
    )
    summary.update(
        {
            "status": "done",
            **metrics,
            **deltas,
            "baseline_all_miou": baseline["metrics"]["all_miou"],
            "baseline_old_miou": baseline["metrics"]["old_miou"],
            "baseline_new_miou": baseline["metrics"]["new_miou"],
            "delta_all_pp": deltas["delta_all"] * 100,
            "delta_old_pp": deltas["delta_old"] * 100,
            "delta_new_pp": deltas["delta_new"] * 100,
            "candidate_count": candidate_count,
            "accepted_count": accepted_count,
            "accepted_ratio": stats.get("accepted_ratio"),
            "mask_counts_match": mask_counts_match,
            "manifest_ok": manifest_ok,
            "result_ok": result_ok,
            "weight_stats_ok": weight_stats_ok,
            "w0_direction_ok": baseline["w0_direction"][row["weighting"]][
                "passed"
            ],
            "per_class_iou": candidate_class_iou,
            "baseline_per_class_iou": baseline["class_iou"],
            "per_class_iou_delta": per_class_iou_delta,
            "weight_mean": stats.get("weight_mean"),
            "weight_std": stats.get("weight_std"),
            "weight_p10": stats.get("weight_p10"),
            "weight_p25": stats.get("weight_p25"),
            "weight_p50": stats.get("weight_p50"),
            "weight_p75": stats.get("weight_p75"),
            "weight_p90": stats.get("weight_p90"),
            "weight_min": stats.get("weight_min"),
            "weight_max": stats.get("weight_max"),
            "weight_extreme_mass_ratio": _weight_extreme_mass_ratio(stats),
            "result_path": str(result_path),
            "result_sha256": file_sha256(result_path),
            "stats_path": str(output_dir / "pseudo_label_stats.json"),
            "manifest_path": str(output_dir / "run_manifest.json"),
        }
    )
    return summary


def evaluate_weighting_gate(weighting, rows):
    failed = []
    if len(rows) != 2 or {row.get("setting") for row in rows} != {
        "overlap",
        "disjoint",
    }:
        return {
            "weighting": weighting,
            "passed": False,
            "failed": ["missing_setting"],
            "mean_delta_all_pp": None,
        }
    if any(row.get("status") != "done" for row in rows):
        failed.append("incomplete")
        mean_delta = None
    else:
        mean_delta = sum(float(row["delta_all_pp"]) for row in rows) / 2
        if mean_delta < W1_MIN_MEAN_ALL_DELTA_PP:
            failed.append("mean_all_delta")
        if any(
            float(row["delta_all_pp"]) < W1_MIN_PER_SETTING_ALL_DELTA_PP
            for row in rows
        ):
            failed.append("per_setting_all_delta")
        if any(float(row["delta_new_pp"]) < W1_MIN_NEW_DELTA_PP for row in rows):
            failed.append("new_class_damage")
        if any(not row.get("w0_direction_ok") for row in rows):
            failed.append("w0_direction")
        if any(not row.get("mask_counts_match") for row in rows):
            failed.append("accepted_mask_mismatch")
        if any(not row.get("manifest_ok") for row in rows):
            failed.append("manifest_mismatch")
        if any(not row.get("result_ok") for row in rows):
            failed.append("result_mismatch")
        if any(not row.get("weight_stats_ok") for row in rows):
            failed.append("weight_stats_invalid")
        if any(
            row.get("weight_extreme_mass_ratio") is None
            or float(row["weight_extreme_mass_ratio"])
            > W1_MAX_EXTREME_MASS_RATIO
            for row in rows
        ):
            failed.append("weight_degenerate")
    return {
        "weighting": weighting,
        "passed": not failed,
        "failed": failed,
        "mean_delta_all_pp": mean_delta,
        "per_setting": {
            row["setting"]: {
                "delta_all_pp": row.get("delta_all_pp"),
                "delta_old_pp": row.get("delta_old_pp"),
                "delta_new_pp": row.get("delta_new_pp"),
            }
            for row in rows
        },
    }


def build_summary(grid_rows, *, checkpoints_root, baselines):
    rows = [
        summarize_candidate_row(
            row,
            checkpoints_root=checkpoints_root,
            baselines=baselines,
        )
        for row in grid_rows
    ]
    gates = {}
    for weighting in ("confidence", "confidence_margin"):
        gates[weighting] = evaluate_weighting_gate(
            weighting,
            [row for row in rows if row["weighting"] == weighting],
        )
    passing = [gate for gate in gates.values() if gate["passed"]]
    if not passing:
        recommendation = "stop"
        preferred = None
    elif len(passing) == 1:
        recommendation = "continue"
        preferred = passing[0]["weighting"]
    else:
        difference = abs(
            gates["confidence"]["mean_delta_all_pp"]
            - gates["confidence_margin"]["mean_delta_all_pp"]
        )
        preferred = (
            "confidence"
            if difference < 0.02
            else max(
                passing,
                key=lambda gate: gate["mean_delta_all_pp"],
            )["weighting"]
        )
        recommendation = "continue"
    if any(row["status"] not in {"done"} for row in rows):
        recommendation = "review"
        preferred = None
    return {
        "schema_version": 1,
        "gate_thresholds": {
            "min_mean_all_delta_pp": W1_MIN_MEAN_ALL_DELTA_PP,
            "min_per_setting_all_delta_pp": W1_MIN_PER_SETTING_ALL_DELTA_PP,
            "min_new_delta_pp": W1_MIN_NEW_DELTA_PP,
            "max_extreme_mass_ratio": W1_MAX_EXTREME_MASS_RATIO,
        },
        "rows": rows,
        "gates": gates,
        "recommendation": recommendation,
        "preferred_weighting": preferred,
    }


def write_outputs(summary, output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")
    md_path = output_base.with_suffix(".md")
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = [
        "name",
        "setting",
        "weighting",
        "status",
        "all_miou",
        "old_miou",
        "new_miou",
        "baseline_all_miou",
        "baseline_old_miou",
        "baseline_new_miou",
        "delta_all_pp",
        "delta_old_pp",
        "delta_new_pp",
        "candidate_count",
        "accepted_count",
        "mask_counts_match",
        "manifest_ok",
        "result_ok",
        "weight_stats_ok",
        "w0_direction_ok",
        "weight_mean",
        "weight_std",
        "weight_extreme_mass_ratio",
        "per_class_iou_delta",
        "result_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            output_row = {key: row.get(key) for key in fields}
            output_row["per_class_iou_delta"] = json.dumps(
                row.get("per_class_iou_delta") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(output_row)

    lines = [
        "# Reliability-Weighted C-RLS W1 Summary",
        "",
        "| setting | weighting | baseline all/old/new | candidate all/old/new | Δall(pp) | Δold(pp) | Δnew(pp) | mask | W0 | manifest | result |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {setting} | {weighting} | {baseline_all_miou}/{baseline_old_miou}/{baseline_new_miou} | "
            "{all_miou}/{old_miou}/{new_miou} | {delta_all_pp} | {delta_old_pp} | {delta_new_pp} | "
            "{mask_counts_match} | {w0_direction_ok} | {manifest_ok} | {result_ok} |".format(
                **{
                    key: row.get(key, "")
                    for key in (
                        "setting",
                        "weighting",
                        "all_miou",
                        "old_miou",
                        "new_miou",
                        "baseline_all_miou",
                        "baseline_old_miou",
                        "baseline_new_miou",
                        "delta_all_pp",
                        "delta_old_pp",
                        "delta_new_pp",
                        "mask_counts_match",
                        "w0_direction_ok",
                        "manifest_ok",
                        "result_ok",
                    )
                }
            )
        )
        if row.get("status") != "done":
            lines.append(
                f"<!-- {row.get('name')}: status={row.get('status')} -->"
            )
    lines.extend(
        [
            "",
            "## Per-class IoU delta",
            "",
        ]
    )
    for row in summary["rows"]:
        if row.get("status") != "done":
            continue
        lines.extend(
            [
                f"### {row['setting']} / {row['weighting']}",
                "",
                "| class | baseline | candidate | delta |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for class_id in sorted(
            row["per_class_iou_delta"],
            key=lambda value: int(value),
        ):
            lines.append(
                f"| {class_id} | {row['baseline_per_class_iou'][class_id]} | "
                f"{row['per_class_iou'][class_id]} | "
                f"{row['per_class_iou_delta'][class_id]} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate",
            "",
            f"- thresholds: `{json.dumps(summary.get('gate_thresholds', {}), sort_keys=True)}`",
            "- accepted-mask 对照：历史 baseline 无逐像素 digest，当前严格核对总体与逐类别 candidate/accepted count。",
            "",
            "```json",
            json.dumps(summary["gates"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            f"- recommendation: `{summary['recommendation']}`",
            f"- preferred_weighting: `{summary['preferred_weighting']}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        default="configs/pseudo_label_weighted_w1_20260719.tsv",
    )
    parser.add_argument(
        "--baselines",
        default="configs/pseudo_label_weighted_w1_baselines_20260719.json",
    )
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument(
        "--output-base",
        default="logs/pseudo_label/weighted_w1_20260719_summary",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rows = read_and_validate_grid(args.grid)
    baselines = load_and_verify_baselines(
        args.baselines,
        repo_root=repo_root,
    )
    summary = build_summary(
        rows,
        checkpoints_root=args.checkpoints_root,
        baselines=baselines,
    )
    paths = write_outputs(summary, args.output_base)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
