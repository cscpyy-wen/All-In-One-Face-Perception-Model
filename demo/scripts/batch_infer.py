#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference: run all images in demo/image/ one by one.

Uses the shared inference modules for consistency with the web demo.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DEMO_DIR))

from inference.config import ensure_dir, load_yaml_as_edict
from inference.image_utils import read_image_rgb
from inference.model import SingleImageMultiTaskModel
from inference.preprocess import MultiTaskPreprocessor
from inference.result_parser import InferenceResult, ResultParser
from inference.visualizer import ResultSaver


def main() -> None:
    cfg_path = DEMO_DIR / "infer.yaml"
    cfg = load_yaml_as_edict(cfg_path)

    device_name = cfg.runtime.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    image_dir = DEMO_DIR / "image"
    output_root = Path(cfg.runtime.output_dir).parent

    print("Loading model ...")
    model = SingleImageMultiTaskModel(cfg, device)
    preprocessor = MultiTaskPreprocessor()
    parser = ResultParser(cfg)

    image_files = sorted(glob.glob(str(image_dir / "*.jpg")))
    print(f"\nFound {len(image_files)} images. Starting batch inference ...\n")

    task_plan = [
        ("recog",      cfg.tasks.recog.task_name),
        ("age",        cfg.tasks.age.task_name),
        ("expression", cfg.tasks.expression.task_name),
        ("attribute",  cfg.tasks.attribute.task_name),
        ("parsing",    cfg.tasks.parsing.task_name),
        ("align",      cfg.tasks.align.task_name),
    ]

    for img_path in image_files:
        name = Path(img_path).stem
        out_dir = output_root / "batch_results" / name
        ensure_dir(out_dir)

        print("=" * 60)
        print(f"  Processing: {name}.jpg  ->  {out_dir}")
        print("=" * 60)

        saver = ResultSaver(out_dir)

        image_rgb = read_image_rgb(img_path)
        prepared = preprocessor.prepare(image_rgb, device)

        results = {}
        parsing_parsed = align_parsed = None

        for short_name, task_name in task_plan:
            tensor = preprocessor.get_tensor(task_name, prepared)
            input_size = preprocessor.get_input_size(task_name)
            raw_output = model.infer_task(tensor, task_name)

            if short_name == "recog":
                parsed = parser.parse_recog(raw_output)
            elif short_name == "age":
                parsed = parser.parse_age(raw_output)
            elif short_name == "expression":
                parsed = parser.parse_expression(raw_output)
            elif short_name == "attribute":
                parsed = parser.parse_attribute(raw_output)
                saver.save_all_attribute_results(parsed)
            elif short_name == "parsing":
                parsed = parser.parse_parsing(raw_output)
                parsing_parsed = parsed
            elif short_name == "align":
                parsed = parser.parse_align(raw_output, input_size)
                align_parsed = parsed

            results[short_name] = InferenceResult(task_name, raw_output, parsed)

        original_rgb = prepared["original_rgb"]

        parsing_overlay = saver.save_parsing_on_original(
            original_rgb, parsing_parsed["mask"],
            parsing_parsed["num_classes"], prepared["wide_rect"],
        )

        landmark_vis = saver.save_align_on_original(
            original_rgb, align_parsed["landmarks"],
            512, prepared["tight_rect"],
        )

        saver.save_demo_summary(
            original_rgb=original_rgb,
            parsing_overlay_rgb=parsing_overlay,
            landmark_vis_rgb=landmark_vis,
            age_result=results["age"].parsed_output,
            expression_result=results["expression"].parsed_output,
            attribute_result=results["attribute"].parsed_output,
        )

        print(f"  Age: {results['age'].parsed_output['estimated_age']:.1f}  "
              f"Exp: {results['expression'].parsed_output['pred_label']}  "
              f"Attrs: {len(results['attribute'].parsed_output['positive_attributes'])}")
        print()

    print("All done!")


if __name__ == "__main__":
    main()
