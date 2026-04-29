#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result visualization and summary image generation.

Produces overlaid parsing masks, landmark drawings, and the combined
summary panel used by both the Gradio web app and CLI batch inference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ensure_dir
from .image_utils import (
    save_rgb,
    resize_to_hw,
    colorize_mask,
    overlay_mask,
    draw_landmarks,
)
from ui.translations import EXPRESSION_ZH, ATTRIBUTE_ZH


class ResultSaver:
    """Generate visual result images and write them to disk."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    # ==================================================================
    # Public: per-task visualizations
    # ==================================================================

    def save_parsing_on_original(
        self,
        original_rgb: np.ndarray,
        mask_512: np.ndarray,
        num_classes: int,
        crop_rect: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Map parsing mask back onto the original image and alpha-blend."""
        vx1, vy1, vx2, vy2 = crop_rect
        crop_w, crop_h = vx2 - vx1, vy2 - vy1
        orig_h, orig_w = original_rgb.shape[:2]

        mask_pil = Image.fromarray(mask_512.astype(np.uint8))
        mask_resized = mask_pil.resize((int(crop_w), int(crop_h)), Image.NEAREST)
        mask_arr = np.asarray(mask_resized)

        mask_rgb = colorize_mask(mask_arr, num_classes)

        canvas = np.zeros_like(original_rgb)
        cx1, cy1 = max(0, int(vx1)), max(0, int(vy1))
        cx2, cy2 = min(orig_w, int(vx2)), min(orig_h, int(vy2))

        mx1 = cx1 - int(vx1)
        my1 = cy1 - int(vy1)
        dst_h = min(cy2 - cy1, mask_rgb.shape[0] - my1)
        dst_w = min(cx2 - cx1, mask_rgb.shape[1] - mx1)

        canvas[cy1:cy1 + dst_h, cx1:cx1 + dst_w] = mask_rgb[my1:my1 + dst_h, mx1:mx1 + dst_w]
        return overlay_mask(original_rgb, canvas)

    def save_align_on_original(
        self,
        original_rgb: np.ndarray,
        landmarks_512: np.ndarray,
        input_size: int,
        crop_rect: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Map landmarks from model space back to original image coordinates."""
        vx1, vy1, vx2, vy2 = crop_rect
        crop_w, crop_h = vx2 - vx1, vy2 - vy1

        pts = landmarks_512.copy()
        pts[:, 0] = pts[:, 0] / input_size * crop_w + vx1
        pts[:, 1] = pts[:, 1] / input_size * crop_h + vy1

        radius = max(2, int(min(original_rgb.shape[:2]) / 150))
        return draw_landmarks(original_rgb, pts, radius=radius)

    # ==================================================================
    # Public: summary panel
    # ==================================================================

    def save_demo_summary(
        self,
        original_rgb: np.ndarray,
        parsing_overlay_rgb: np.ndarray,
        landmark_vis_rgb: np.ndarray,
        age_result: dict[str, Any],
        expression_result: dict[str, Any],
        attribute_result: dict[str, Any],
    ) -> None:
        """Create a combined summary image: original + parsing + landmarks + info bar."""
        h_orig, w_orig = original_rgb.shape[:2]

        pad, gap = 16, 8
        img_h, bottom_h = 300, 80

        disp_w = min(int(w_orig * img_h / h_orig), img_h * 2)

        original = resize_to_hw(original_rgb, disp_w, img_h)
        parsing = resize_to_hw(parsing_overlay_rgb, disp_w, img_h)
        landmark = resize_to_hw(landmark_vis_rgb, disp_w, img_h)

        total_w = pad * 2 + disp_w * 3 + gap * 2
        total_h = pad + img_h + gap + bottom_h + pad

        bottom_panel = self._make_bottom_panel(
            width=total_w - pad * 2,
            height=bottom_h,
            age_result=age_result,
            expression_result=expression_result,
            attribute_result=attribute_result,
        )

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        y, x = pad, pad
        for img in (original, parsing, landmark):
            canvas.paste(Image.fromarray(img.astype(np.uint8)), (x, y))
            x += disp_w + gap
        y += img_h + gap
        canvas.paste(Image.fromarray(bottom_panel.astype(np.uint8)), (pad, y))

        save_rgb(self.output_dir / "演示结果总览.png", np.asarray(canvas))

    # ==================================================================
    # Public: text output
    # ==================================================================

    def save_all_attribute_results(self, parsed: dict[str, Any]) -> None:
        """Write all attribute scores to a text file."""
        all_scores = parsed.get("all_scores", [])
        attr_names = parsed.get("attribute_names", [])

        txt_path = self.output_dir / "全部属性预测结果.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Moeface attribute prediction results\n")
            f.write("=" * 48 + "\n\n")
            for i, score in enumerate(all_scores):
                name = attr_names[i] if i < len(attr_names) else f"attr_{i}"
                f.write(f"{name}\t{float(score):.6f}\n")

    # ==================================================================
    # Internal: bottom info panel
    # ==================================================================

    def _make_bottom_panel(
        self,
        width: int,
        height: int,
        age_result: dict[str, Any],
        expression_result: dict[str, Any],
        attribute_result: dict[str, Any],
    ) -> np.ndarray:
        panel = Image.new("RGB", (width, height), (245, 245, 248))
        draw = ImageDraw.Draw(panel)
        font = self._get_font(22)

        attrs = self._format_demo_attributes(attribute_result)

        est_age = age_result.get("estimated_age")
        age_text = self._format_age(est_age) if est_age is not None else "未知"

        exp_zh = EXPRESSION_ZH.get(
            expression_result.get("pred_label", "未知"), "未知"
        )

        items = [
            ("年龄", age_text),
            ("性别", attrs["gender"]),
            ("表情", exp_zh),
            ("发色", attrs["hair"]),
            ("刘海", attrs["bangs"]),
            ("化妆", attrs["makeup"]),
            ("眼镜", attrs["eye"]),
        ]

        label_color = (120, 120, 130)
        value_color = (30, 30, 30)
        sep_color = (210, 210, 215)

        col_w = width // len(items)
        y_label = height // 2 - 18
        y_value = height // 2 + 6

        for i, (label, value) in enumerate(items):
            cx = col_w * i + col_w // 2
            lw = draw.textbbox((0, 0), label, font=font)[2]
            draw.text((cx - lw // 2, y_label), label, fill=label_color, font=font)

            vw = draw.textbbox((0, 0), value, font=font)[2]
            draw.text((cx - vw // 2, y_value), value, fill=value_color, font=font)

            if i > 0:
                sx = col_w * i
                draw.line([(sx, 16), (sx, height - 16)], fill=sep_color, width=1)

        draw.line([(0, 0), (width, 0)], fill=(200, 200, 205), width=2)
        return np.asarray(panel)

    # ==================================================================
    # Internal: attribute formatting
    # ==================================================================

    @staticmethod
    def _format_age(est_age: float) -> str:
        if est_age >= 25:
            return f"{est_age - 5:.0f}岁"
        if est_age >= 22:
            return f"{est_age - 2:.0f}岁"
        return f"{est_age:.0f}岁"

    @staticmethod
    def _get_attr_score(attribute_result: dict, target_name: str) -> float | None:
        names = attribute_result.get("attribute_names", [])
        scores = attribute_result.get("all_scores", [])
        for name, score in zip(names, scores):
            if name == target_name:
                return float(score)
        return None

    @classmethod
    def _format_demo_attributes(cls, attribute_result: dict) -> dict[str, str]:
        male_score = cls._get_attr_score(attribute_result, "Male")
        gender = "男性" if (male_score is not None and male_score >= 0.5) else "女性"

        hair_scores = [
            (n, cls._get_attr_score(attribute_result, n) or 0.0)
            for n in ("Black_Hair", "Blond_Hair", "Brown_Hair")
        ]
        hair_name = max(hair_scores, key=lambda x: x[1])[0]
        hair_color = ATTRIBUTE_ZH.get(hair_name, hair_name)

        def _attr_bool(name, yes_text, no_text):
            s = cls._get_attr_score(attribute_result, name)
            return yes_text if (s is not None and s >= 0.5) else no_text

        return {
            "gender": gender,
            "hair": hair_color,
            "makeup": _attr_bool("Heavy_Makeup", "浓妆", "未化妆"),
            "eye": _attr_bool("Eyeglasses", "戴眼镜", "未戴眼镜"),
            "bangs": _attr_bool("Bangs", "有", "无"),
        }

    # ==================================================================
    # Internal: font loading
    # ==================================================================

    @staticmethod
    def _find_chinese_font() -> str | None:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    @classmethod
    def _get_font(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_path = cls._find_chinese_font()
        if font_path is not None:
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
        return ImageFont.load_default()
