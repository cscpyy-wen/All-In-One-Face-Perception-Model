#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core inference pipeline orchestration.

Initializes the model once at import time, then exposes
``run_inference()`` and ``format_results()`` for the Gradio UI.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import DEMO_DIR, load_yaml_as_edict
from .image_utils import colorize_mask, draw_landmarks, overlay_mask
from .model import SingleImageMultiTaskModel
from .preprocess import MultiTaskPreprocessor
from .result_parser import ResultParser
from .visualizer import ResultSaver
from ui.translations import EXPRESSION_ZH, ATTRIBUTE_ZH
from ui.styles import WAIT_HTML


# ============================================================================
# Global model initialization (loaded once at startup)
# ============================================================================

print("=" * 60)
print("  Moeface 多任务人脸分析系统")
print("  正在加载模型，请稍候...")
print("=" * 60)

CFG_PATH = DEMO_DIR / "infer.yaml"
cfg = load_yaml_as_edict(CFG_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  推理设备: {DEVICE}")

model = SingleImageMultiTaskModel(cfg, DEVICE)
preprocessor = MultiTaskPreprocessor()
result_parser = ResultParser(cfg)

HAS_CUDA = torch.cuda.is_available()
print(f"  CUDA: {'可用' if HAS_CUDA else '不可用（使用 CPU）'}")
print("=" * 60)
print("  模型加载完成！")
print("=" * 60)

# Ordered task plan
TASK_PLAN = [
    ("recog",      cfg.tasks.recog.task_name),
    ("age",        cfg.tasks.age.task_name),
    ("expression", cfg.tasks.expression.task_name),
    ("attribute",  cfg.tasks.attribute.task_name),
    ("parsing",    cfg.tasks.parsing.task_name),
    ("align",      cfg.tasks.align.task_name),
]


# ============================================================================
# Core inference
# ============================================================================

def run_inference(image_rgb: np.ndarray) -> dict[str, Any] | None:
    """Run all 6 tasks on a single image and return all raw + visual results."""
    if image_rgb is None:
        return None

    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)
    elif image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]

    prepared = preprocessor.prepare(image_rgb, DEVICE)
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

    saver = ResultSaver(DEMO_DIR / "__tmp_app__")
    original_rgb = prepared["original_rgb"]

    parsing_on_orig = saver.save_parsing_on_original(
        original_rgb, parsing_parsed["mask"], parsing_parsed["num_classes"],
        prepared["wide_rect"],
    )
    landmark_on_orig = saver.save_align_on_original(
        original_rgb, align_parsed["landmarks"], 512, prepared["tight_rect"],
    )

    wide_crop_512 = prepared.get("image_512_rgb")
    if wide_crop_512 is not None:
        mask_rgb_crop = colorize_mask(parsing_parsed["mask"],
                                      parsing_parsed["num_classes"])
        parsing_on_crop = overlay_mask(wide_crop_512, mask_rgb_crop)
    else:
        parsing_on_crop = parsing_on_orig

    tight_crop_512 = prepared.get("image_512_tight_rgb")
    if tight_crop_512 is not None:
        landmark_on_crop = draw_landmarks(tight_crop_512,
                                          align_parsed["landmarks"], radius=4)
    else:
        landmark_on_crop = landmark_on_orig

    results.update({
        "_parsing_on_orig": parsing_on_orig,
        "_landmark_on_orig": landmark_on_orig,
        "_parsing_on_crop": parsing_on_crop,
        "_landmark_on_crop": landmark_on_crop,
        "_original_rgb": original_rgb,
    })

    return results


# ============================================================================
# Result formatting for Gradio
# ============================================================================

def format_results(results: dict[str, Any] | None) -> tuple:
    """Convert raw inference results into Gradio-compatible outputs."""
    if results is None:
        return None, None, None, WAIT_HTML, None

    original = results["_original_rgb"]
    parsing_on_crop = results["_parsing_on_crop"]
    landmark_on_crop = results["_landmark_on_crop"]

    est_age = results["age"]["estimated_age"]
    age_text = _format_display_age(est_age)

    exp_en = results["expression"]["pred_label"]
    exp_zh = EXPRESSION_ZH.get(exp_en, exp_en)
    exp_prob = results["expression"]["probabilities"]
    exp_idx = results["expression"]["pred_index"]
    exp_conf = exp_prob[exp_idx] * 100

    attr_scores = results["attribute"]["all_scores"]
    attr_names = results["attribute"]["attribute_names"]

    def _score(name):
        for n, s in zip(attr_names, attr_scores):
            if n == name:
                return s
        return 0.0

    male_score = _score("Male")
    gender = "男" if male_score >= 0.5 else "女"
    gender_conf = max(male_score, 1 - male_score) * 100

    hair_name = max(
        [(n, _score(n)) for n in ("Black_Hair", "Blond_Hair", "Brown_Hair")],
        key=lambda x: x[1],
    )[0]
    hair_zh = ATTRIBUTE_ZH.get(hair_name, hair_name)

    summary_html = _build_summary_html(
        age_text, gender, gender_conf, exp_zh, exp_conf, hair_zh,
        makeup="浓妆" if _score("Heavy_Makeup") >= 0.5 else "无",
        glasses="戴" if _score("Eyeglasses") >= 0.5 else "无",
        bangs="有" if _score("Bangs") >= 0.5 else "无",
        smiling="是" if _score("Smiling") >= 0.5 else "否",
    )

    detail_html = _build_detail_html(
        expression_labels=list(cfg.labels.expression),
        expression_probs=exp_prob,
        expression_idx=exp_idx,
        attr_names=attr_names,
        attr_scores=attr_scores,
    )

    all_html = summary_html + "\n" + detail_html

    saver = ResultSaver(DEMO_DIR / "__tmp_app__")
    saver.save_demo_summary(
        original_rgb=original,
        parsing_overlay_rgb=results["_parsing_on_orig"],
        landmark_vis_rgb=results["_landmark_on_orig"],
        age_result=results["age"],
        expression_result=results["expression"],
        attribute_result=results["attribute"],
    )
    summary_path = DEMO_DIR / "__tmp_app__" / "演示结果总览.png"
    summary_arr = np.asarray(Image.open(str(summary_path))) if summary_path.exists() else None
    summary_file = str(summary_path) if summary_path.exists() else None

    return parsing_on_crop, landmark_on_crop, summary_arr, all_html, summary_file


def inference_image_upload(image):
    """Handle image upload / webcam snapshot via the Gradio interface."""
    if image is None:
        return None, None, None, WAIT_HTML, None
    t0 = time.perf_counter()
    results = run_inference(image)
    elapsed = time.perf_counter() - t0
    print(f"[Inference] {elapsed:.2f}s")
    return format_results(results)


# ============================================================================
# Internal: HTML builders
# ============================================================================

def _format_display_age(est_age: float) -> str:
    if est_age >= 25:
        return f"{est_age - 5:.0f}"
    if est_age >= 22:
        return f"{est_age - 2:.0f}"
    return f"{est_age:.0f}"


def _build_summary_html(age, gender, gender_conf, exp_zh, exp_conf,
                         hair, makeup, glasses, bangs, smiling) -> str:
    return f"""
    <div class="attr-card">
      <div class="attr-grid">
        <div class="attr-item"><div class="attr-label">年龄</div><div class="attr-value">{age} 岁</div></div>
        <div class="attr-item"><div class="attr-label">性别</div><div class="attr-value">{gender} <span class="conf">{gender_conf:.0f}%</span></div></div>
        <div class="attr-item"><div class="attr-label">表情</div><div class="attr-value">{exp_zh} <span class="conf">{exp_conf:.0f}%</span></div></div>
        <div class="attr-item"><div class="attr-label">发色</div><div class="attr-value">{hair}</div></div>
        <div class="attr-item"><div class="attr-label">化妆</div><div class="attr-value">{makeup}</div></div>
        <div class="attr-item"><div class="attr-label">眼镜</div><div class="attr-value">{glasses}</div></div>
        <div class="attr-item"><div class="attr-label">刘海</div><div class="attr-value">{bangs}</div></div>
        <div class="attr-item"><div class="attr-label">微笑</div><div class="attr-value">{smiling}</div></div>
      </div>
    </div>
    """


def _build_detail_html(
    expression_labels: list[str],
    expression_probs: list[float],
    expression_idx: int,
    attr_names: list[str],
    attr_scores: list[float],
) -> str:
    exp_bars = ""
    for i, (label, prob) in enumerate(zip(expression_labels, expression_probs)):
        zh = EXPRESSION_ZH.get(label, label)
        pct = prob * 100
        active = "active" if i == expression_idx else ""
        exp_bars += f"""
        <div class="bar-row {active}">
          <span class="bar-label">{zh}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>"""

    sorted_attrs = sorted(zip(attr_names, attr_scores), key=lambda x: x[1], reverse=True)
    attr_bars = ""
    for name, score in sorted_attrs[:12]:
        zh = ATTRIBUTE_ZH.get(name, name)
        pct = score * 100
        active = "active" if score >= 0.5 else ""
        attr_bars += f"""
        <div class="bar-row {active}">
          <span class="bar-label">{zh}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>"""

    return f"""
    <div class="detail-row">
      <div class="detail-col">
        <div class="detail-title">表情分布</div>
        {exp_bars}
      </div>
      <div class="detail-col">
        <div class="detail-title">属性置信度 (Top 12)</div>
        {attr_bars}
      </div>
    </div>
    """
