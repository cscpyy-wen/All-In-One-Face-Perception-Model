# All-In-One Multi-Task Face Perception Model

本项目是一个基于 Vision Transformer 的多任务人脸感知模型。可同时完成 **6 项人脸分析任务**：

| 任务 |
|---|---|---|
| 人脸识别 (Face Recognition) 
| 年龄估计 (Age Estimation) 
| 表情识别 (Expression Recognition)
| 属性分析 (Attribute Analysis)|
| 人脸解析 (Face Parsing)
| 关键点检测 (Landmark Detection)|

```
Moeface/
├── Moeface_project/          # 模型核心代码
│   ├── core/
│   │   ├── model/            # 模型定义（backbone、heads、loss）
│   │   ├── data/             # 数据加载
│   │   ├── evaluator/        # 评估模块
│   │   ├── solver/           # 训练求解器
│   │   └── transform/        # 数据增强
│   ├── train.py              # 训练入口
│   └── test.py               # 测试入口
│
├── demo/                     # 交互式 Web 演示
│   ├── app.py                # Gradio 演示入口
│   ├── inference/            # 推理引擎模块
│   │   ├── config.py         # 配置管理
│   │   ├── model.py          # 模型加载
│   │   ├── preprocess.py     # 人脸检测 + 预处理
│   │   ├── result_parser.py  # 结果解析
│   │   ├── visualizer.py     # 可视化
│   │   ├── pipeline.py       # 推理管线
│   │   └── stream.py         # 实时视频流
│   ├── ui/                   # UI 组件
│   │   ├── styles.py         # CSS 样式
│   │   └── translations.py   # 中文翻译
│   ├── scripts/              # 命令行脚本
│   ├── infer.yaml            # 推理配置
│   ├── checkpoint/           # 模型权重（需下载）
│   └── image/                # 测试图片
│
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n moeface python=3.9 -y
conda activate moeface

# 安装 PyTorch（CUDA 12.4）
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 下载模型权重

运行推理需要下载两个模型文件：

| 文件 | 大小 | 放置位置 |
|---|---|---|
| `the_writer.pth.tar` | ~1.8 GB | `demo/checkpoint/` |
| `FaRL-Base-Patch16-LAIONFace20M-ep64.pth` | ~650 MB | `Moeface_project/pretrain/` |

```bash
# 创建目录
mkdir -p demo/checkpoint Moeface_project/pretrain

# 下载（替换为实际的下载链接）
# wget <download_url> -O demo/checkpoint/the_writer.pth.tar
# wget <download_url> -O Moeface_project/pretrain/FaRL-Base-Patch16-LAIONFace20M-ep64.pth
```

> 模型权重下载链接请参见 [Releases](../../releases) 页面。

### 3. 启动 Web Demo

```bash
cd demo
python app.py
```

浏览器会自动打开 `http://localhost:7860`，支持三种输入模式：

- **上传图片** — 拖拽或点击上传
- **摄像头拍照** — 点击摄像头图标拍照后分析
- **实时视频** — 打开摄像头实时分析（~3 FPS）

### 4. 命令行推理

```bash
cd demo

# 单张图片推理
python scripts/infer.py --infer_config infer.yaml

# 批量推理（处理 image/ 目录下所有图片）
python scripts/batch_infer.py

# 直接推理（无人脸检测，直接 resize 整张图）
python scripts/infer_direct.py --infer_config infer.yaml
```

## 技术细节

### 推理流程

```
输入图片 → InsightFace 人脸检测 → 双分辨率裁剪
                                         │
                        ┌────────────────┴────────────────┐
                        │                                 │
                   112×112 裁剪                       512×512 裁剪
                        │                                 │
              ┌─────────┼─────────┐             ┌─────────┼─────────┐
              ▼         ▼         ▼             ▼         ▼         ▼
           Recognition  Age   Expression    Parsing  Landmark
            Attribute
```

### 支持的任务与标签

**表情 (7 类):** 中性、高兴、伤心、惊讶、害怕、厌恶、愤怒

**属性 (40 项):** 性别、发色、是否戴眼镜、是否微笑、是否有刘海 等 40 项 CelebA 属性

**人脸解析 (19 类):** 背景、脖子、脸、衣服、眉毛、眼睛、虹膜、鼻子、嘴唇、头发、眼镜、帽子、耳环、项链

**关键点:** 98 个面部关键点（基于 WFLW 数据集）

## 依赖说明

| 包 | 用途 |
|---|---|
| PyTorch >= 2.6 | 深度学习框架 |
| Gradio >= 4.0 | Web 演示界面 |
| InsightFace | 人脸检测 |
| timm | ViT 模型组件 |
| OpenCV | 图像处理 |
| easydict | 配置管理 |

完整列表见 `requirements.txt`。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 致谢

- [FaRL](https://github.com/FacePerceiver/FaRL) — 面部表征学习预训练模型
- [InsightFace](https://github.com/deepinsight/insightface) — 人脸检测与分析工具箱
- [Faceptor](https://github.com/lxq1000/Faceptor.git)— 多任务人脸感知模型
