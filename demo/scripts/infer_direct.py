#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct inference — no face detection, just resize and feed to model.

Output format is identical to infer.py for comparison.
Reuses shared inference modules from the demo package.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DEMO_DIR))

from inference.config import ensure_dir, load_yaml_as_edict
from inference.image_utils import (
    read_image_rgb, save_rgb, resize_rgb, resize_to_hw,
    normalize, softmax_np, sigmoid_np,
    colorize_mask, overlay_mask, draw_landmarks, PARSING_COLORS,
    CLASSIFICATION_MEAN, CLASSIFICATION_STD, DENSE_MEAN, DENSE_STD,
)
from inference.model import SingleImageMultiTaskModel
from inference.result_parser import ResultParser
from inference.visualizer import ResultSaver


# ============================================================================
# Direct preprocessor (no face detection)
# ============================================================================

class DirectPreprocessor:
    """No face detection — just resize the whole image."""

    def prepare(self, image_rgb: np.ndarray, device: torch.device) -> Dict[str, Any]:
        img_112 = resize_rgb(image_rgb, 112)
        img_512 = resize_rgb(image_rgb, 512)

        tensor_112 = normalize(img_112, CLASSIFICATION_MEAN, CLASSIFICATION_STD).unsqueeze(0).to(device)
        tensor_512 = normalize(img_512, DENSE_MEAN, DENSE_STD).unsqueeze(0).to(device)

        return {
            "original_rgb": image_rgb,
            "image_512_rgb": img_512,
            "tensor_112": tensor_112,
            "tensor_512": tensor_512,
        }

    def get_tensor(self, task_name: str, prepared: Dict[str, Any]) -> torch.Tensor:
        if task_name.startswith(("recog_", "age_", "biattr_", "affect_")):
            return prepared["tensor_112"]
        if task_name.startswith(("parsing_", "align_")):
            return prepared["tensor_512"]
        raise ValueError(f"Unknown task: {task_name}")

    def get_input_size(self, task_name: str) -> int:
        if task_name.startswith(("recog_", "age_", "biattr_", "affect_")):
            return 112
        if task_name.startswith(("parsing_", "align_")):
            return 512
        raise ValueError(f"Unknown task: {task_name}")


# ============================================================================
# Direct result saver (simplified — no crop rect mapping)
# ============================================================================

class DirectResultSaver(ResultSaver):
    """Override visualization methods for the no-detection case."""

    def save_parsing_on_original(self, original_rgb, mask_512, num_classes, crop_rect=None):
        orig_h, orig_w = original_rgb.shape[:2]
        mask_pil = Image.fromarray(mask_512.astype(np.uint8))
        mask_resized = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        mask_arr = np.asarray(mask_resized)
        mask_rgb = colorize_mask(mask_arr, num_classes)
        return overlay_mask(original_rgb, mask_rgb)

    def save_align_on_original(self, original_rgb, landmarks_512, input_size, crop_rect=None):
        orig_h, orig_w = original_rgb.shape[:2]
        pts = landmarks_512.copy()
        pts[:, 0] = pts[:, 0] / input_size * orig_w
        pts[:, 1] = pts[:, 1] / input_size * orig_h
        return draw_landmarks(original_rgb, pts, radius=max(2, int(min(orig_h, orig_w) / 150)))


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Direct inference (no face detection)")
    parser.add_argument("--infer_config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml_as_edict(args.infer_config)
    output_dir = Path(cfg.runtime.output_dir).parent / "direct_results"
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")

    model = SingleImageMultiTaskModel(cfg, device)
    preprocessor = DirectPreprocessor()
    result_parser = ResultParser(cfg)
    saver = DirectResultSaver(output_dir)

    image_dir = DEMO_DIR / "image"
    image_files = sorted(glob.glob(str(image_dir / "*.jpg")))
    print(f"Found {len(image_files)} images. Starting direct inference ...\n")

    for img_path in image_files:
        name = Path(img_path).stem
        out_dir = output_dir / name
        ensure_dir(out_dir)
        saver.output_dir = out_dir

        image_rgb = read_image_rgb(img_path)
        prepared = preprocessor.prepare(image_rgb, device)

        task_plan = [
            ("recog",      cfg.tasks.recog.task_name),
            ("age",        cfg.tasks.age.task_name),
            ("expression", cfg.tasks.expression.task_name),
            ("attribute",  cfg.tasks.attribute.task_name),
            ("parsing",    cfg.tasks.parsing.task_name),
            ("align",      cfg.tasks.align.task_name),
        ]

        results: Dict[str, Any] = {}
        parsing_vis = align_vis = None

        for short_name, task_name in task_plan:
            tensor = preprocessor.get_tensor(task_name, prepared)
            input_size = preprocessor.get_input_size(task_name)
            raw_output = model.infer_task(tensor, task_name)

            if short_name == "recog":
                parsed = result_parser.parse_recog(raw_output)
            elif short_name == "age":
                parsed = result_parser.parse_age(raw_output)
            elif short_name == "expression":
                parsed = result_parser.parse_expression(raw_output)
            elif short_name == "attribute":
                parsed = result_parser.parse_attribute(raw_output)
                saver.save_all_attribute_results(parsed)
            elif short_name == "parsing":
                parsed = result_parser.parse_parsing(raw_output)
                parsing_vis = saver.save_parsing_on_original(
                    prepared["original_rgb"], parsed["mask"], parsed["num_classes"])
            elif short_name == "align":
                parsed = result_parser.parse_align(raw_output, input_size)
                align_vis = saver.save_align_on_original(
                    prepared["original_rgb"], parsed["landmarks"], input_size)

            results[short_name] = {"task_name": task_name, "parsed": parsed}

        if parsing_vis is not None and align_vis is not None:
            saver.save_demo_summary(
                original_rgb=prepared["original_rgb"],
                parsing_overlay_rgb=parsing_vis,
                landmark_vis_rgb=align_vis,
                age_result=results["age"]["parsed"],
                expression_result=results["expression"]["parsed"],
                attribute_result=results["attribute"]["parsed"],
            )

        attr = results["attribute"]["parsed"]
        male_s = [s for n, s in zip(attr["attribute_names"], attr["all_scores"]) if n == "Male"][0]
        print(f"{name}: Age={results['age']['parsed']['estimated_age']:.1f}  "
              f"Exp={results['expression']['parsed']['pred_label']}  Male={male_s:.3f}")

    print(f"\nAll done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
