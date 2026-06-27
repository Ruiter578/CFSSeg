import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


VALID_STRATEGIES = {
    "off",
    "fixed",
    "batch_global",
    "batch_class",
    "artifact_class",
}


@dataclass
class PseudoLabelConfig:
    strategy: str = "off"
    fixed_confidence: float = 0.7
    quantile: float = 0.7
    min_conf: float = 0.0
    max_conf: float = 1.0
    min_pixels: int = 1
    shrinkage: float = 0.0
    margin_min: float = 0.0
    threshold_artifact: Optional[str] = None
    stats: bool = False
    artifact_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class PseudoLabelCandidates:
    scores: torch.Tensor
    labels: torch.Tensor
    margins: torch.Tensor
    mask: torch.Tensor


@dataclass
class PseudoLabelBatchStats:
    strategy: str
    candidate_count: int
    accepted_count: int
    thresholds: Dict[str, float]
    per_class_candidates: Dict[str, int]
    per_class_accepted: Dict[str, int]
    fallbacks: Dict[str, str] = field(default_factory=dict)


@dataclass
class PseudoLabelResult:
    labels: torch.Tensor
    mask: torch.Tensor
    stats: PseudoLabelBatchStats


def resolve_pseudo_label_strategy(use_pseudo_label, pseudo_label_strategy):
    if not use_pseudo_label:
        return "off"
    strategy = pseudo_label_strategy or "fixed"
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported pseudo_label_strategy: {strategy}")
    return strategy


def validate_pseudo_label_config(config: PseudoLabelConfig):
    if config.strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported pseudo_label_strategy: {config.strategy}")
    if not 0.0 <= config.quantile <= 1.0:
        raise ValueError("pseudo_label_quantile must be in [0, 1].")
    if not 0.0 <= config.min_conf <= 1.0:
        raise ValueError("pseudo_label_min_conf must be in [0, 1].")
    if not 0.0 <= config.max_conf <= 1.0:
        raise ValueError("pseudo_label_max_conf must be in [0, 1].")
    if config.min_conf > config.max_conf:
        raise ValueError("pseudo_label_min_conf must be <= pseudo_label_max_conf.")
    if config.min_pixels < 1:
        raise ValueError("pseudo_label_min_pixels must be >= 1.")
    if config.shrinkage < 0:
        raise ValueError("pseudo_label_shrinkage must be >= 0.")
    if config.margin_min < 0:
        raise ValueError("pseudo_label_margin_min must be >= 0.")


def config_from_opts(opts):
    strategy = resolve_pseudo_label_strategy(
        getattr(opts, "use_pseudo_label", False),
        getattr(opts, "pseudo_label_strategy", None),
    )
    config = PseudoLabelConfig(
        strategy=strategy,
        fixed_confidence=float(getattr(opts, "pseudo_label_confidence", 0.7)),
        quantile=float(getattr(opts, "pseudo_label_quantile", 0.7)),
        min_conf=float(getattr(opts, "pseudo_label_min_conf", 0.0)),
        max_conf=float(getattr(opts, "pseudo_label_max_conf", 1.0)),
        min_pixels=int(getattr(opts, "pseudo_label_min_pixels", 1)),
        shrinkage=float(getattr(opts, "pseudo_label_shrinkage", 0.0)),
        margin_min=float(getattr(opts, "pseudo_label_margin_min", 0.0)),
        threshold_artifact=getattr(opts, "pseudo_label_threshold_artifact", None),
        stats=bool(getattr(opts, "pseudo_label_stats", False)),
    )
    validate_pseudo_label_config(config)
    return config


def _as_tensor_output(model_output):
    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise ValueError("Teacher output tuple/list is empty.")
        model_output = model_output[0]
    if not torch.is_tensor(model_output):
        raise TypeError(f"Teacher output must be a tensor, got {type(model_output)!r}.")
    if model_output.ndim != 4:
        raise ValueError(f"Teacher output must be 4D, got shape {tuple(model_output.shape)}.")
    return model_output


def _to_nchw(logits: torch.Tensor, expected_classes: int):
    channels_first = logits.shape[1] == expected_classes
    channels_last = logits.shape[-1] == expected_classes
    if channels_first and channels_last:
        raise ValueError(
            "Teacher output layout is ambiguous because both dim=1 and dim=-1 "
            f"match expected_classes={expected_classes}."
        )
    if channels_first:
        return logits
    if channels_last:
        return logits.permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        "Teacher output class dimension does not match "
        f"expected_classes={expected_classes}: shape={tuple(logits.shape)}."
    )


def extract_teacher_probabilities(model_output, *, loss_type: str, expected_classes: int):
    logits = _to_nchw(_as_tensor_output(model_output), int(expected_classes))
    if loss_type == "bce_loss":
        return torch.sigmoid(logits)
    if loss_type in {"ce_loss", "focal_loss"}:
        return torch.softmax(logits, dim=1)
    raise ValueError(f"Unsupported loss_type for pseudo labeling: {loss_type}")


def resize_probabilities_to_labels(probabilities: torch.Tensor, labels: torch.Tensor):
    if probabilities.shape[-2:] == labels.shape[-2:]:
        return probabilities
    return F.interpolate(
        probabilities,
        labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _old_class_mask(pred_labels: torch.Tensor, old_class_ids: Iterable[int]):
    mask = torch.zeros_like(pred_labels, dtype=torch.bool)
    for class_id in old_class_ids:
        mask |= pred_labels == int(class_id)
    return mask


def compute_pseudo_label_candidates(probabilities, labels, old_class_ids):
    if probabilities.ndim != 4:
        raise ValueError("probabilities must be NCHW.")
    if labels.ndim != 3:
        raise ValueError("labels must be NHW.")
    if probabilities.shape[0] != labels.shape[0] or probabilities.shape[-2:] != labels.shape[-2:]:
        raise ValueError(
            "probabilities and labels must share batch and spatial shape: "
            f"{tuple(probabilities.shape)} vs {tuple(labels.shape)}"
        )

    topk = torch.topk(probabilities, k=min(2, probabilities.shape[1]), dim=1)
    scores = topk.values[:, 0]
    pred_labels = topk.indices[:, 0]
    if probabilities.shape[1] > 1:
        margins = topk.values[:, 0] - topk.values[:, 1]
    else:
        margins = topk.values[:, 0]

    background_mask = labels == 0
    valid_mask = labels != 255
    old_mask = _old_class_mask(pred_labels, old_class_ids)
    candidate_mask = background_mask & valid_mask & old_mask
    return PseudoLabelCandidates(
        scores=scores,
        labels=pred_labels,
        margins=margins,
        mask=candidate_mask,
    )


def _clip_threshold(value, config: PseudoLabelConfig):
    return float(max(config.min_conf, min(config.max_conf, float(value))))


def _quantile_or_none(values: torch.Tensor, quantile: float):
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.detach().float().cpu(), float(quantile)).item())


def _class_counts(candidates: PseudoLabelCandidates, old_class_ids):
    counts = {}
    for class_id in old_class_ids:
        class_mask = candidates.mask & (candidates.labels == int(class_id))
        counts[str(int(class_id))] = int(class_mask.sum().item())
    return counts


def resolve_thresholds_with_fallbacks(candidates, config: PseudoLabelConfig, old_class_ids):
    validate_pseudo_label_config(config)
    old_class_ids = [int(class_id) for class_id in old_class_ids]

    if config.strategy == "artifact_class":
        if not config.artifact_thresholds:
            raise ValueError("artifact_class requires loaded artifact_thresholds.")
        missing = [
            str(class_id)
            for class_id in old_class_ids
            if str(class_id) not in config.artifact_thresholds
        ]
        if missing:
            raise ValueError(
                f"artifact_thresholds missing entries for class ids: {missing}."
            )
        return (
            {str(class_id): float(config.artifact_thresholds[str(class_id)]) for class_id in old_class_ids},
            {str(class_id): "artifact" for class_id in old_class_ids},
        )

    if config.strategy == "fixed":
        fixed = _clip_threshold(config.fixed_confidence, config)
        return (
            {str(class_id): fixed for class_id in old_class_ids},
            {str(class_id): "fixed" for class_id in old_class_ids},
        )

    candidate_scores = candidates.scores[candidates.mask]
    global_raw = _quantile_or_none(candidate_scores, config.quantile)
    global_fallback = "fixed_empty" if global_raw is None else "none"
    global_threshold = _clip_threshold(
        config.fixed_confidence if global_raw is None else global_raw,
        config,
    )

    if config.strategy == "batch_global":
        return (
            {str(class_id): global_threshold for class_id in old_class_ids},
            {str(class_id): global_fallback for class_id in old_class_ids},
        )

    if config.strategy != "batch_class":
        raise ValueError(f"Strategy {config.strategy} does not resolve thresholds.")

    thresholds = {}
    fallbacks = {}
    for class_id in old_class_ids:
        class_mask = candidates.mask & (candidates.labels == int(class_id))
        class_scores = candidates.scores[class_mask]
        raw = _quantile_or_none(class_scores, config.quantile)
        if raw is None or class_scores.numel() < config.min_pixels:
            thresholds[str(class_id)] = global_threshold
            fallbacks[str(class_id)] = "global"
            continue
        if config.shrinkage > 0:
            n = float(class_scores.numel())
            raw = (n * raw + config.shrinkage * global_threshold) / (n + config.shrinkage)
        thresholds[str(class_id)] = _clip_threshold(raw, config)
        fallbacks[str(class_id)] = "none"
    return thresholds, fallbacks


def resolve_thresholds(candidates, config: PseudoLabelConfig, old_class_ids):
    thresholds, _ = resolve_thresholds_with_fallbacks(candidates, config, old_class_ids)
    return thresholds


def apply_pseudo_labels(labels, candidates, thresholds, margin_min: float, fallbacks=None):
    pseudo_labels = labels.clone()
    accepted = candidates.mask & (candidates.margins >= float(margin_min))
    for class_key, threshold in thresholds.items():
        class_id = int(class_key)
        class_mask = candidates.labels == class_id
        accepted &= ~(class_mask & (candidates.scores < float(threshold)))

    pseudo_labels[accepted] = candidates.labels[accepted].to(pseudo_labels.dtype)

    per_class_candidates = _class_counts(candidates, thresholds.keys())
    per_class_accepted = {}
    for class_key in thresholds:
        class_id = int(class_key)
        per_class_accepted[class_key] = int((accepted & (candidates.labels == class_id)).sum().item())

    stats = PseudoLabelBatchStats(
        strategy="",
        candidate_count=int(candidates.mask.sum().item()),
        accepted_count=int(accepted.sum().item()),
        thresholds={str(key): float(value) for key, value in thresholds.items()},
        per_class_candidates=per_class_candidates,
        per_class_accepted=per_class_accepted,
        fallbacks={str(key): str(value) for key, value in (fallbacks or {}).items()},
    )
    return PseudoLabelResult(labels=pseudo_labels, mask=accepted, stats=stats)


def load_threshold_artifact(
    path,
    *,
    dataset,
    task,
    step,
    setting,
    old_class_ids,
    teacher_sha256=None,
    quantile=None,
    min_conf=None,
    max_conf=None,
    min_pixels=None,
    shrinkage=None,
    loss_type=None,
    random_seed=None,
    max_batches=None,
):
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "dataset": dataset,
        "task": task,
        "step": int(step),
        "setting": setting,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(
                f"Pseudo-label threshold artifact {artifact_path} has {key}={artifact.get(key)!r}, "
                f"expected {value!r}."
            )
    artifact_old = [int(item) for item in artifact.get("old_class_ids", [])]
    expected_old = [int(item) for item in old_class_ids]
    if artifact_old != expected_old:
        raise ValueError(
            f"Pseudo-label threshold artifact old_class_ids={artifact_old}, expected {expected_old}."
        )
    if teacher_sha256 and artifact.get("teacher_sha256") != teacher_sha256:
        raise ValueError("Pseudo-label threshold artifact teacher_sha256 does not match.")
    optional_expected = {
        "quantile": quantile,
        "min_conf": min_conf,
        "max_conf": max_conf,
        "min_pixels": min_pixels,
        "shrinkage": shrinkage,
        "loss_type": loss_type,
        "random_seed": random_seed,
        "max_batches": max_batches,
    }
    for key, value in optional_expected.items():
        if value is None or key not in artifact:
            continue
        artifact_value = artifact.get(key)
        if isinstance(value, float):
            if abs(float(artifact_value) - float(value)) > 1e-12:
                raise ValueError(
                    f"Pseudo-label threshold artifact {key}={artifact_value!r}, "
                    f"expected {value!r}."
                )
        elif artifact_value != value:
            raise ValueError(
                f"Pseudo-label threshold artifact {key}={artifact_value!r}, "
                f"expected {value!r}."
            )

    thresholds = {}
    classes = artifact.get("classes", {})
    global_threshold = float(artifact.get("global_threshold", 0.0))
    for class_id in expected_old:
        class_record = classes.get(str(class_id), {})
        thresholds[str(class_id)] = float(class_record.get("final_threshold", global_threshold))
    return thresholds


class PseudoLabeler:
    def __init__(
        self,
        config: PseudoLabelConfig,
        *,
        old_class_ids,
        loss_type,
        expected_classes,
    ):
        validate_pseudo_label_config(config)
        self.config = config
        self.old_class_ids = [int(class_id) for class_id in old_class_ids]
        self.loss_type = loss_type
        self.expected_classes = int(expected_classes)

    @property
    def enabled(self):
        return self.config.strategy != "off"

    def apply(self, model_output, labels):
        probabilities = extract_teacher_probabilities(
            model_output,
            loss_type=self.loss_type,
            expected_classes=self.expected_classes,
        )
        probabilities = resize_probabilities_to_labels(probabilities, labels)
        candidates = compute_pseudo_label_candidates(
            probabilities,
            labels,
            self.old_class_ids,
        )
        thresholds, fallbacks = resolve_thresholds_with_fallbacks(
            candidates,
            self.config,
            self.old_class_ids,
        )
        result = apply_pseudo_labels(
            labels,
            candidates,
            thresholds,
            self.config.margin_min,
            fallbacks,
        )
        result.stats.strategy = self.config.strategy
        return result
