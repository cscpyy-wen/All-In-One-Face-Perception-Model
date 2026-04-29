#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moeface 多任务人脸分析系统 — Gradio Demo
答辩演示用，支持图片上传 / 摄像头快照 / 实时视频流
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Import modular components
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(CURRENT_DIR))

from ui.styles import CSS, HEADER_HTML, FOOTER_HTML, WAIT_HTML
from inference.pipeline import inference_image_upload
from inference.stream import StreamingProcessor

# Global streaming processor instance
_stream_proc = StreamingProcessor()

# Example images (auto-discover jpg files in demo/image/)
EXAMPLE_IMGS = sorted(
    str(p) for p in (CURRENT_DIR / "image").glob("*.jpg")
    if p.name.startswith(("test", "1", "2", "0"))
)


# ============================================================================
# Gradio UI
# ============================================================================

def build_app():
    with gr.Blocks(
        title="All-In-One 多任务人脸感知模型",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            font=gr.themes.GoogleFont("Noto Sans SC"),
        ),
        css=CSS,
    ) as app:

        # ---- Header ----
        gr.HTML(HEADER_HTML)

        # ---- Source toggle buttons ----
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1, min_width=400):
                with gr.Row():
                    btn_upload = gr.Button("📁 上传图片", variant="secondary", size="lg")
                    btn_webcam = gr.Button("📷 摄像头拍照", variant="secondary", size="lg")
                    btn_stream = gr.Button("📹 实时视频", variant="secondary", size="lg")
            with gr.Column(scale=1):
                pass

        # ---- Static mode (upload + webcam snapshot) ----
        with gr.Row() as static_row:

            # == LEFT: Input ==
            with gr.Column(scale=2, min_width=340, elem_classes=["col-divider"]):

                with gr.Group(visible=True) as upload_panel:
                    input_upload = gr.Image(
                        label="拖拽、粘贴或点击上传图片",
                        type="numpy",
                        sources=["upload", "clipboard"],
                        height=380,
                        elem_classes=["input-wrap"],
                        show_download_button=False,
                        show_share_button=False,
                    )

                with gr.Group(visible=False) as webcam_panel:
                    input_webcam = gr.Image(
                        label="摄像头",
                        type="numpy",
                        sources=["webcam"],
                        height=380,
                        elem_classes=["input-wrap"],
                        show_download_button=False,
                        show_share_button=False,
                    )

                current_input = gr.State("upload")

                btn_run = gr.Button(
                    "🔍 开始分析",
                    variant="primary",
                    size="lg",
                    elem_classes=["run-btn"],
                )

                gr.Examples(
                    examples=[[p] for p in EXAMPLE_IMGS],
                    inputs=input_upload,
                    label="示例图片（点击加载）",
                )

            # == RIGHT: Results ==
            with gr.Column(scale=3, min_width=560):

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-title">人脸解析</div>')
                        img_parsing = gr.Image(
                            label=None, type="numpy",
                            elem_classes=["result-img"],
                            show_download_button=True, show_share_button=False,
                        )
                    with gr.Column():
                        gr.HTML('<div class="section-title">关键点检测</div>')
                        img_landmark = gr.Image(
                            label=None, type="numpy",
                            elem_classes=["result-img"],
                            show_download_button=True, show_share_button=False,
                        )

                results_html = gr.HTML(WAIT_HTML)

                img_summary = gr.Image(
                    visible=False, type="numpy",
                    show_download_button=False, show_share_button=False,
                )
                btn_download = gr.DownloadButton(
                    "📥 下载综合结果总览",
                    visible=False,
                    variant="secondary",
                    size="lg",
                )

        # ---- Streaming panel (hidden by default) ----
        with gr.Row(visible=False) as stream_row:
            with gr.Column():
                stream_input = gr.Image(
                    sources=["webcam"], type="numpy", streaming=True,
                    label="摄像头输入", height=500,
                    show_download_button=False, show_share_button=False,
                )
            with gr.Column():
                stream_output = gr.Image(
                    type="numpy", streaming=True,
                    label="实时分析结果", height=500,
                    show_download_button=False, show_share_button=False,
                )

        # ================================================================
        # Event handlers
        # ================================================================

        def _run(mode, upload_img, webcam_img):
            img = webcam_img if mode == "webcam" else upload_img
            parsing, landmark, summary, html, summary_file = inference_image_upload(img)
            return (
                parsing, landmark, summary, html,
                gr.update(value=summary_file, visible=summary_file is not None),
            )

        btn_run.click(
            fn=_run,
            inputs=[current_input, input_upload, input_webcam],
            outputs=[img_parsing, img_landmark, img_summary, results_html, btn_download],
        )

        stream_input.stream(
            fn=lambda frame: _stream_proc.submit(frame),
            inputs=[stream_input],
            outputs=[stream_output],
        )

        # ---- Mode switching ----
        def switch_to_upload():
            _stream_proc.reset()
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                "upload",
            )

        def switch_to_webcam():
            _stream_proc.reset()
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                "webcam",
            )

        def switch_to_stream():
            _stream_proc.reset()
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                "stream",
            )

        btn_upload.click(switch_to_upload,
                         outputs=[static_row, stream_row, upload_panel,
                                  webcam_panel, current_input])
        btn_webcam.click(switch_to_webcam,
                         outputs=[static_row, stream_row, upload_panel,
                                  webcam_panel, current_input])
        btn_stream.click(switch_to_stream,
                         outputs=[static_row, stream_row, upload_panel,
                                  webcam_panel, current_input])

        # ---- Footer ----
        gr.HTML(FOOTER_HTML)

    return app


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
