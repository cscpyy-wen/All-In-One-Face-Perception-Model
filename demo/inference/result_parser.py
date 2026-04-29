#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result parsing for each inference task.

Converts raw model outputs (logits, maps, embeddings) into structured
dictionaries with human-readable labels and probability distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Ensure Moeface_project is on sys.path before importing core.*
from . import config  # noqa: F401

from core.model.geometry import denormalize_points

from .image_utils import softmax_np, sigmoid_np


# ============================================================================
# Data container
# ============================================================================

@dataclass
class InferenceResult:
    """Container for a single task's raw and parsed outputs."""
    task_name: str
    raw_output: Any
    parsed_output: dict[str, Any]


# ============================================================================
# Result parser
# ============================================================================

class ResultParser:
    """Parse raw model outputs for each of the six tasks."""

    def __init__(self, cfg) -> None:
        self.expression_labels: list[str] = list(cfg.labels.expression)
        self.attribute_labels: list[str] = list(cfg.labels.attribute)
        self.parsing_labels: list[str] = list(cfg.labels.parsing)

    # ------------------------------------------------------------------
    # Recognition — returns L2-normalized embedding
    # ------------------------------------------------------------------

    def parse_recog(self, output: Any) -> dict[str, Any]:
        feat = self._to_numpy(output).reshape(-1)
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        return {
            "embedding_dim": int(feat.shape[0]),
            "embedding_preview": feat[:16].tolist(),
            "embedding_full": feat.tolist(),
        }

    # ------------------------------------------------------------------
    # Age estimation — DLDL soft distribution
    # ------------------------------------------------------------------

    def parse_age(self, output: Any) -> dict[str, Any]:
        logits = self._to_numpy(output).reshape(-1)
        probs = sigmoid_np(logits)
        probs = probs / (np.sum(probs) + 1e-12)
        rank = np.arange(len(probs), dtype=np.float32)
        estimated_age = float(np.sum(probs * rank))
        return {
            "estimated_age": estimated_age,
            "top5_bins": np.argsort(-probs)[:5].tolist(),
            "distribution": probs.tolist(),
        }

    # ------------------------------------------------------------------
    # Expression recognition — softmax over 7 classes
    # ------------------------------------------------------------------

    def parse_expression(self, output: Any) -> dict[str, Any]:
        logits = self._to_numpy(output).reshape(-1)
        probs = softmax_np(logits)
        idx = int(np.argmax(probs))
        label = self.expression_labels[idx] if idx < len(self.expression_labels) else f"class_{idx}"
        return {
            "pred_index": idx,
            "pred_label": label,
            "probabilities": probs.tolist(),
        }

    # ------------------------------------------------------------------
    # Binary attributes — sigmoid per attribute (CelebA 40-dim)
    # ------------------------------------------------------------------

    def parse_attribute(self, output: Any, threshold: float = 0.5) -> dict[str, Any]:
        logits = self._to_numpy(output).reshape(-1)
        probs = sigmoid_np(logits)
        positives = []
        for i, p in enumerate(probs):
            label = self.attribute_labels[i] if i < len(self.attribute_labels) else f"attr_{i}"
            if p >= threshold:
                positives.append({"index": i, "label": label, "score": float(p)})
        positives.sort(key=lambda x: x["score"], reverse=True)
        return {
            "threshold": threshold,
            "positive_attributes": positives,
            "all_scores": probs.tolist(),
            "attribute_names": list(self.attribute_labels),
        }

    # ------------------------------------------------------------------
    # Face parsing — argmax over class dimension
    # ------------------------------------------------------------------

    def parse_parsing(self, output: Any) -> dict[str, Any]:
        logits = self._to_numpy(output)
        if logits.ndim == 4:
            logits = logits[0]
        if logits.ndim != 3:
            raise ValueError(f"Unexpected parsing output shape: {logits.shape}")
        pred_mask = np.argmax(logits, axis=-1).astype(np.int32)
        return {
            "mask": pred_mask,
            "num_classes": int(logits.shape[-1]),
            "labels": self.parsing_labels,
        }

    # ------------------------------------------------------------------
    # Landmark alignment — denormalize coordinates
    # ------------------------------------------------------------------

    def parse_align(self, output: Any, input_size: int) -> dict[str, Any]:
        points = output.get("landmark") if isinstance(output, dict) and "landmark" in output else output
        if not torch.is_tensor(points):
            points = torch.as_tensor(points, dtype=torch.float32)
        if points.ndim == 2:
            points = points.unsqueeze(0)
        points = denormalize_points(points, input_size, input_size)
        points = points.detach().cpu().float().numpy()[0]
        return {
            "num_points": int(points.shape[0]),
            "landmarks": points.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().float().numpy()
        return np.asarray(x)
