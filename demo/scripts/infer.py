#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for single-image batch inference.

Usage:
    python scripts/infer.py --infer_config infer.yaml

This script reuses the modular inference components so that all logic
lives in one place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch

# Ensure the demo root directory is importable
SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DEMO_DIR))

from inference.config import ensure_dir, load_yaml_as_edict
from inference.image_utils import read_image_rgb
from inference.model import SingleImageMultiTaskModel
from inference.preprocess import MultiTaskPreprocessor
from inference.result_parser import InferenceResult, ResultParser
from inference.visualizer import ResultSaver


# ============================================================================
# Batch inference runner
# ============================================================================

class MultiTaskInferenceRunner:
    """Run all six tasks on a single image and save visual results."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        device_name = cfg.runtime.device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        self.preprocessor = MultiTaskPreprocessor()
        self.parser = ResultParser(cfg)
        self.saver = ResultSaver(cfg.runtime.output_dir)
        self.model = SingleImageMultiTaskModel(cfg, self.device)

    @torch.no_grad()
    def run(self) -> Dict[str, InferenceResult]:
        image_rgb = read_image_rgb(self.cfg.paths.image)
        prepared = self.preprocessor.prepare(image_rgb, self.device)

        task_plan = [
            ("recog",      self.cfg.tasks.recog.task_name),
            ("age",        self.cfg.tasks.age.task_name),
            ("expression", self.cfg.tasks.expression.task_name),
            ("attribute",  self.cfg.tasks.attribute.task_name),
            ("parsing",    self.cfg.tasks.parsing.task_name),
            ("align",      self.cfg.tasks.align.task_name),
        ]

        results: Dict[str, InferenceResult] = {}
        parsing_vis = align_vis = None

        for short_name, task_name in task_plan:
            tensor = self.preprocessor.get_tensor(task_name, prepared)
            input_size = self.preprocessor.get_input_size(task_name)
            raw_output = self.model.infer_task(tensor, task_name)

            if short_name == "recog":
                parsed = self.parser.parse_recog(raw_output)
            elif short_name == "age":
                parsed = self.parser.parse_age(raw_output)
            elif short_name == "expression":
                parsed = self.parser.parse_expression(raw_output)
            elif short_name == "attribute":
                parsed = self.parser.parse_attribute(raw_output)
                self.saver.save_all_attribute_results(parsed)
            elif short_name == "parsing":
                parsed = self.parser.parse_parsing(raw_output)
                parsing_vis = {
                    "overlay_rgb": self.saver.save_parsing_on_original(
                        prepared["original_rgb"], parsed["mask"],
                        parsed["num_classes"], prepared["wide_rect"],
                    ),
                }
            elif short_name == "align":
                parsed = self.parser.parse_align(raw_output, input_size)
                align_vis = {
                    "landmark_vis_rgb": self.saver.save_align_on_original(
                        prepared["original_rgb"], parsed["landmarks"],
                        input_size, prepared["tight_rect"],
                    ),
                }
            else:
                raise ValueError(f"Unsupported task: {short_name}")

            results[short_name] = InferenceResult(
                task_name=task_name,
                raw_output=raw_output,
                parsed_output=parsed,
            )

        if parsing_vis is not None and align_vis is not None:
            self.saver.save_demo_summary(
                original_rgb=prepared["original_rgb"],
                parsing_overlay_rgb=parsing_vis["overlay_rgb"],
                landmark_vis_rgb=align_vis["landmark_vis_rgb"],
                age_result=results["age"].parsed_output,
                expression_result=results["expression"].parsed_output,
                attribute_result=results["attribute"].parsed_output,
            )

        return results


# ============================================================================
# CLI
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-image six-task inference")
    parser.add_argument("--infer_config", type=str, required=True,
                        help="Path to single-image inference yaml config")
    return parser


def print_summary(results: Dict[str, InferenceResult]) -> None:
    sep = "=" * 80
    print(sep)
    print("Inference finished")
    print(sep)
    print(f"[Recognition] embedding_dim = {results['recog'].parsed_output['embedding_dim']}")
    print(f"[Age] estimated_age = {results['age'].parsed_output['estimated_age']:.2f}")
    exp_out = results["expression"].parsed_output
    print(f"[Expression] {exp_out['pred_label']} (idx={exp_out['pred_index']})")
    print(f"[Attributes] positive_count = {len(results['attribute'].parsed_output['positive_attributes'])}")
    print(f"[Parsing] num_classes = {results['parsing'].parsed_output['num_classes']}")
    print(f"[Alignment] num_points = {results['align'].parsed_output['num_points']}")
    print(sep)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    cfg = load_yaml_as_edict(args.infer_config)
    ensure_dir(cfg.runtime.output_dir)

    runner = MultiTaskInferenceRunner(cfg)
    results = runner.run()
    print_summary(results)


if __name__ == "__main__":
    main()
