#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face detection and multi-task preprocessing pipeline.

Detects faces via InsightFace, crops wide/tight regions, normalizes them
to the sizes required by each downstream task (112x112 for classification,
512x512 for dense prediction).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .image_utils import (
    resize_rgb,
    normalize,
    CLASSIFICATION_MEAN, CLASSIFICATION_STD,
    DENSE_MEAN, DENSE_STD,
)


class MultiTaskPreprocessor:
    """Prepare a raw RGB image for multi-task inference."""

    def __init__(self) -> None:
        self._face_app = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        image_rgb: np.ndarray,
        device: torch.device,
        cached_bbox: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Detect face, crop, and produce all required tensors."""
        prepared: dict[str, Any] = {"original_rgb": image_rgb}

        bbox = cached_bbox if cached_bbox is not None else self._detect_faces(image_rgb)

        if bbox is not None:
            wide, wide_rect = self._crop_face(image_rgb, bbox, expand=1.8)
            tight, tight_rect = self._crop_face(image_rgb, bbox, expand=1.0)

            prepared.update({
                "face_detected": True,
                "cached_bbox": bbox,
                "image_112_rgb": resize_rgb(wide, 112),
                "image_512_rgb": resize_rgb(wide, 512),
                "image_512_tight_rgb": resize_rgb(tight, 512),
                "tensor_112_wide": self._to_tensor(wide, 112, True, device),
                "tensor_112_tight": self._to_tensor(tight, 112, True, device),
                "tensor_512_wide": self._to_tensor(wide, 512, False, device),
                "tensor_512_tight": self._to_tensor(tight, 512, False, device),
                "wide_rect": wide_rect,
                "tight_rect": tight_rect,
            })
        else:
            h, w = image_rgb.shape[:2]
            prepared.update({
                "face_detected": False,
                "cached_bbox": None,
                "image_112_rgb": resize_rgb(image_rgb, 112),
                "image_512_rgb": resize_rgb(image_rgb, 512),
                "image_512_tight_rgb": resize_rgb(image_rgb, 512),
                "tensor_112_wide": self._to_tensor(image_rgb, 112, True, device),
                "tensor_112_tight": self._to_tensor(image_rgb, 112, True, device),
                "tensor_512_wide": self._to_tensor(image_rgb, 512, False, device),
                "tensor_512_tight": self._to_tensor(image_rgb, 512, False, device),
                "wide_rect": (0, 0, w, h),
                "tight_rect": (0, 0, w, h),
            })

        return prepared

    def get_tensor(self, task_name: str, prepared: dict[str, Any]) -> torch.Tensor:
        """Select the correct pre-processed tensor for a given task."""
        if task_name.startswith(("biattr_", "affect_", "recog_")):
            return prepared["tensor_112_wide"]
        if task_name.startswith("age_"):
            return prepared["tensor_112_tight"]
        if task_name.startswith("parsing_"):
            return prepared["tensor_512_wide"]
        if task_name.startswith("align_"):
            return prepared["tensor_512_tight"]
        raise ValueError(f"Unknown task: {task_name}")

    def get_input_size(self, task_name: str) -> int:
        """Return the input resolution for a given task."""
        if task_name.startswith(("recog_", "age_", "biattr_", "affect_")):
            return 112
        if task_name.startswith(("parsing_", "align_")):
            return 512
        raise ValueError(f"Unknown task: {task_name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_faces(self, image_rgb: np.ndarray) -> np.ndarray | None:
        app = self._get_face_app()
        faces = app.get(image_rgb)
        if not faces:
            return None
        return faces[0].bbox.astype(float)

    def _get_face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._face_app = FaceAnalysis(providers=providers)
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app

    @staticmethod
    def _crop_face(
        image_rgb: np.ndarray,
        bbox: np.ndarray,
        expand: float,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_rgb.shape[:2]
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_size = max(fw, fh) * expand / 2.0

        vx1, vy1 = cx - half_size, cy - half_size
        vx2, vy2 = cx + half_size, cy + half_size

        c_x1, c_y1 = int(vx1), int(vy1)
        c_x2, c_y2 = int(vx2), int(vy2)

        pad_l = max(0, -c_x1)
        pad_t = max(0, -c_y1)
        pad_r = max(0, c_x2 - img_w)
        pad_b = max(0, c_y2 - img_h)

        if pad_l or pad_t or pad_r or pad_b:
            padded = np.pad(image_rgb,
                            ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                            mode="constant", constant_values=0)
            c_x1 += pad_l; c_y1 += pad_t
            c_x2 += pad_l; c_y2 += pad_t
        else:
            padded = image_rgb

        return padded[c_y1:c_y2, c_x1:c_x2], (vx1, vy1, vx2, vy2)

    @staticmethod
    def _to_tensor(
        image_rgb: np.ndarray,
        size: int,
        classification: bool,
        device: torch.device,
    ) -> torch.Tensor:
        resized = resize_rgb(image_rgb, size)
        mean = CLASSIFICATION_MEAN if classification else DENSE_MEAN
        std = CLASSIFICATION_STD if classification else DENSE_STD
        return normalize(resized, mean, std).unsqueeze(0).to(device)
