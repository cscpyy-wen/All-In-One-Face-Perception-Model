#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time video streaming processor.

Runs face analysis in a background thread and returns annotated frames
to the Gradio streaming interface with minimal latency.
"""

from __future__ import annotations

import time
import threading
from typing import Any

import cv2
import numpy as np

from .image_utils import colorize_mask, draw_landmarks
from .pipeline import (
    model, preprocessor, result_parser, TASK_PLAN, DEVICE, cfg,
)
from ui.translations import EXPRESSION_ZH


class StreamingProcessor:
    """Decoupled processor: background thread runs inference while
    Gradio immediately returns the latest annotated frame."""

    def __init__(self, redetect_interval: int = 5) -> None:
        self.cached_bbox: np.ndarray | None = None
        self.frame_count = 0
        self.redetect_interval = redetect_interval

        self._lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._latest_result: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.cached_bbox = None
        self.frame_count = 0
        with self._lock:
            self._pending_frame = None
            self._latest_result = None

    def submit(self, frame: np.ndarray) -> np.ndarray | None:
        """Called by Gradio stream — stores frame, returns latest result."""
        if frame is None:
            return None
        self._ensure_thread()
        with self._lock:
            self._pending_frame = frame.copy()
            result = self._latest_result
        return result if result is not None else frame

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def _inference_loop(self) -> None:
        while self._running:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None

            if frame is not None:
                t0 = time.perf_counter()
                result = self._process_one(frame)
                elapsed = time.perf_counter() - t0
                with self._lock:
                    self._latest_result = result
                print(f"[Stream] {elapsed * 1000:.0f}ms frame #{self.frame_count}",
                      flush=True)
            else:
                time.sleep(0.005)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_one(self, frame: np.ndarray) -> np.ndarray | None:
        if frame is None:
            return None

        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        self.frame_count += 1

        need_detect = (
            self.cached_bbox is None
            or self.frame_count % self.redetect_interval == 0
        )

        if need_detect:
            prepared = preprocessor.prepare(frame, DEVICE)
        else:
            prepared = preprocessor.prepare(frame, DEVICE, cached_bbox=self.cached_bbox)

        self.cached_bbox = prepared.get("cached_bbox")

        if not prepared.get("face_detected", False):
            return frame

        results: dict[str, Any] = {}
        parsing_parsed = align_parsed = None

        for short_name, task_name in TASK_PLAN:
            tensor = preprocessor.get_tensor(task_name, prepared)
            input_size = preprocessor.get_input_size(task_name)
            raw_output = model.infer_task(tensor, task_name)

            if short_name == "parsing":
                parsed = result_parser.parse_parsing(raw_output)
                parsing_parsed = parsed
            elif short_name == "align":
                parsed = result_parser.parse_align(raw_output, input_size)
                align_parsed = parsed
            else:
                parsed = getattr(result_parser, f"parse_{short_name}")(raw_output)

            results[short_name] = parsed

        annotated = frame.copy()
        orig_h, orig_w = annotated.shape[:2]

        _draw_parsing_overlay(annotated, parsing_parsed, prepared, orig_w, orig_h)
        _draw_landmark_overlay(annotated, align_parsed, prepared, orig_h, orig_w)
        _draw_info_bar(annotated, results, orig_h, orig_w)

        return annotated


# ============================================================================
# Overlay drawing helpers
# ============================================================================

def _draw_parsing_overlay(
    annotated: np.ndarray,
    parsing_parsed: dict | None,
    prepared: dict,
    orig_w: int,
    orig_h: int,
) -> None:
    if parsing_parsed is None:
        return

    vx1, vy1, vx2, vy2 = prepared["wide_rect"]
    crop_w, crop_h = vx2 - vx1, vy2 - vy1

    mask_resized = cv2.resize(
        parsing_parsed["mask"].astype(np.uint8),
        (int(crop_w), int(crop_h)),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_rgb = colorize_mask(mask_resized, parsing_parsed["num_classes"])

    cx1, cy1 = max(0, int(vx1)), max(0, int(vy1))
    cx2, cy2 = min(orig_w, int(vx2)), min(orig_h, int(vy2))

    mx1 = cx1 - int(vx1)
    my1 = cy1 - int(vy1)
    dst_h = min(cy2 - cy1, mask_rgb.shape[0] - my1)
    dst_w = min(cx2 - cx1, mask_rgb.shape[1] - mx1)

    roi = annotated[cy1:cy1 + dst_h, cx1:cx1 + dst_w].astype(np.float32)
    msk = mask_rgb[my1:my1 + dst_h, mx1:mx1 + dst_w].astype(np.float32)
    blended = (roi * 0.55 + msk * 0.45).clip(0, 255).astype(np.uint8)
    annotated[cy1:cy1 + dst_h, cx1:cx1 + dst_w] = blended


def _draw_landmark_overlay(
    annotated: np.ndarray,
    align_parsed: dict | None,
    prepared: dict,
    orig_h: int,
    orig_w: int,
) -> None:
    if align_parsed is None:
        return

    vx1, vy1, vx2, vy2 = prepared["tight_rect"]
    crop_w, crop_h = vx2 - vx1, vy2 - vy1

    pts = align_parsed["landmarks"].copy()
    pts[:, 0] = pts[:, 0] / 512 * crop_w + vx1
    pts[:, 1] = pts[:, 1] / 512 * crop_h + vy1

    radius = max(2, int(min(orig_h, orig_w) / 200))
    for x, y in pts.astype(int):
        cv2.circle(annotated, (x, y), radius, (0, 255, 0), -1)


def _draw_info_bar(
    annotated: np.ndarray,
    results: dict,
    orig_h: int,
    orig_w: int,
) -> None:
    attr_scores = results["attribute"]["all_scores"]
    attr_names = results["attribute"]["attribute_names"]

    male_score = smile_score = glasses_score = 0.0
    for n, s in zip(attr_names, attr_scores):
        if n == "Male":
            male_score = s
        elif n == "Smiling":
            smile_score = s
        elif n == "Eyeglasses":
            glasses_score = s

    gender = "Male" if male_score >= 0.5 else "Female"
    smile = "Yes" if smile_score >= 0.5 else "No"
    glasses = "Yes" if glasses_score >= 0.5 else "No"
    exp_en = results["expression"]["pred_label"]

    est_age = results["age"]["estimated_age"]
    if est_age >= 25:
        age_text = f"{est_age - 5:.0f}"
    elif est_age >= 22:
        age_text = f"{est_age - 2:.0f}"
    else:
        age_text = f"{est_age:.0f}"

    bar_h = max(80, int(orig_h * 0.1))
    overlay_bar = annotated[:bar_h].copy()
    cv2.rectangle(overlay_bar, (0, 0), (orig_w, bar_h), (20, 20, 40), -1)
    alpha = 0.7
    annotated[:bar_h] = cv2.addWeighted(overlay_bar, alpha, annotated[:bar_h], 1 - alpha, 0)

    font_scale = max(0.7, min(orig_w, orig_h) / 700)
    thickness = max(2, int(font_scale * 2.5))
    line_h = int(font_scale * 35)
    x_pos = max(10, int(orig_w * 0.03))

    lines = [
        f"Age: {age_text}   Gender: {gender}",
        f"Expression: {exp_en}   Smile: {smile}   Glasses: {glasses}",
    ]
    y_pos = int(bar_h * 0.35)
    for line in lines:
        cv2.putText(annotated, line, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(annotated, line, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        y_pos += line_h
