#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSS styles and HTML templates for the Gradio web UI.

Keeps all visual design artifacts in one place so the main app.py
can focus purely on layout and wiring.
"""

# ============================================================================
# Main stylesheet
# ============================================================================

CSS = """
/* ---- Global ---- */
.gradio-container { max-width: 1400px !important; margin: auto; }
#logo { text-align: center; padding: 20px 0 6px; }
#logo h1 { font-size: 28px; font-weight: 700; margin: 0; color: #1a1a2e; }
#logo p  { font-size: 13px; color: #888; margin: 4px 0 0; }
.tag { display: inline-block; background: #eef2ff; color: #4f46e5;
       font-size: 11px; padding: 2px 8px; border-radius: 10px; margin: 0 2px; }

/* ---- Source toggle bar ---- */
.src-bar { display: flex; justify-content: center; gap: 16px; margin-bottom: 14px; }
.src-btn { padding: 10px 32px; font-size: 16px; font-weight: 600;
           border: 2px solid #e5e7eb; border-radius: 10px;
           background: #fff; color: #6b7280; cursor: pointer;
           transition: all .2s; display: flex; align-items: center; gap: 8px; }
.src-btn:hover { border-color: #818cf8; color: #4f46e5; }
.src-btn.active { background: #4f46e5; color: #fff; border-color: #4f46e5; }

/* ---- Input area ---- */
.input-wrap { border-radius: 12px; border: 1px solid #e5e7eb;
              overflow: hidden; background: #fafbfc; }

/* ---- Action button ---- */
.run-btn { width: 100%; font-size: 18px !important; font-weight: 600 !important;
           padding: 12px 0 !important; border-radius: 10px !important; }

/* ---- Divider between columns ---- */
.col-divider { border-right: 2px solid #f0f0f5; padding-right: 24px; }

/* ---- Result section titles ---- */
.section-title { font-size: 15px; font-weight: 600; color: #374151;
                 padding: 10px 0 6px; border-bottom: 2px solid #e5e7eb;
                 margin-bottom: 8px; }

/* ---- Results images ---- */
.result-img { border-radius: 10px; overflow: hidden; border: 1px solid #eee; }

/* ---- Attribute card ---- */
.attr-card { background: #fff; border: 1px solid #e5e7eb;
             border-radius: 12px; padding: 16px 12px; }
.attr-grid { display: grid; grid-template-columns: repeat(4, 1fr);
             gap: 10px; text-align: center; }
.attr-item { padding: 8px 2px; border-radius: 8px; background: #f9fafb; }
.attr-label { font-size: 12px; color: #9ca3af; margin-bottom: 2px; }
.attr-value { font-size: 16px; font-weight: 600; color: #1f2937; }
.attr-value .conf { font-size: 11px; color: #9ca3af; font-weight: 400; }

/* ---- Bar charts ---- */
.detail-row { display: flex; gap: 24px; }
.detail-col { flex: 1; }
.detail-title { font-size: 14px; font-weight: 600; color: #374151;
                margin-bottom: 8px; padding-bottom: 4px;
                border-bottom: 2px solid #e5e7eb; }
.bar-row { display: flex; align-items: center; gap: 8px;
           padding: 3px 0; font-size: 13px; }
.bar-row.active .bar-label { font-weight: 600; color: #1f2937; }
.bar-label { width: 64px; flex-shrink: 0; color: #6b7280; text-align: right; }
.bar-track { flex: 1; height: 10px; background: #f3f4f6;
             border-radius: 5px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px;
            background: linear-gradient(90deg, #818cf8, #6366f1);
            transition: width .4s ease; }
.bar-row.active .bar-fill {
    background: linear-gradient(90deg, #4f46e5, #3730a3); }
.bar-pct { width: 42px; text-align: right; color: #9ca3af; font-size: 12px; }

/* ---- Webcam capture button override ---- */
.webcam-btn-wrap { display: flex; justify-content: center; margin: 10px 0; }
.capture-btn { padding: 14px 48px !important; font-size: 20px !important;
               font-weight: 700 !important; border-radius: 14px !important; }

/* ---- Footer ---- */
#footer { text-align: center; padding: 20px 0 10px;
          border-top: 1px solid #eee; margin-top: 16px; }
#footer .author { font-size: 15px; font-weight: 600; color: #374151;
                  margin-bottom: 2px; }
#footer .affil { font-size: 12px; color: #9ca3af; }
footer { display: none !important; }
.small { display: none !important; }
"""

# ============================================================================
# Placeholder HTML (shown before analysis)
# ============================================================================

WAIT_HTML = (
    '<div class="waiting">'
    '上传图片或点击摄像头图标拍照，然后点击「开始分析」'
    '</div>'
)

# ============================================================================
# Header HTML
# ============================================================================

HEADER_HTML = """
<div id="logo">
  <h1>All-In-One 多任务人脸感知模型</h1>
  <p>
    <span class="tag">年龄估计</span>
    <span class="tag">性别属性</span>
    <span class="tag">表情识别</span>
    <span class="tag">人脸解析</span>
    <span class="tag">关键点检测</span>
  </p>
</div>
"""

# ============================================================================
# Footer HTML
# ============================================================================

FOOTER_HTML = """
<div id="footer">
  <p class="author">Zhihao Wen (文之豪)</p>
  <p class="affil">
    Undergraduate, Central South University &nbsp;&middot;&nbsp;
    Ph.D. Student, Institute of Computing Technology, Chinese Academy of Sciences
  </p>
</div>
"""
