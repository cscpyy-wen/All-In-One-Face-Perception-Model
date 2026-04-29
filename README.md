# All-In-One Multi-Task Face Perception Model

A unified multi-task face perception model based on Vision Transformer. A single forward pass performs **6 face analysis tasks** simultaneously:

| Task |
|---|
| Face Recognition |
| Age Estimation |
| Expression Recognition |
| Attribute Analysis |
| Face Parsing |
| Landmark Detection |

```
Moeface/
├── Moeface_project/          # Core model code
│   ├── core/
│   │   ├── model/            # Model definition (backbone, heads, loss)
│   │   ├── data/             # Data loading
│   │   ├── evaluator/        # Evaluation modules
│   │   ├── solver/           # Training solver
│   │   └── transform/        # Data augmentation
│   ├── train.py              # Training entry
│   └── test.py               # Testing entry
│
├── demo/                     # Interactive web demo
│   ├── app.py                # Gradio demo entry
│   ├── inference/            # Inference engine modules
│   │   ├── config.py         # Configuration management
│   │   ├── model.py          # Model loading
│   │   ├── preprocess.py     # Face detection & preprocessing
│   │   ├── result_parser.py  # Result parsing
│   │   ├── visualizer.py     # Visualization
│   │   ├── pipeline.py       # Inference pipeline
│   │   └── stream.py         # Real-time video stream
│   ├── ui/                   # UI components
│   │   ├── styles.py         # CSS styles
│   │   └── translations.py   # Chinese translations
│   ├── scripts/              # CLI scripts
│   ├── infer.yaml            # Inference config
│   ├── checkpoint/           # Model weights (download required)
│   └── image/                # Test images
│
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Getting Started

### 1. Environment Setup

```bash
# Create conda environment
conda create -n moeface python=3.9 -y
conda activate moeface

# Install PyTorch (CUDA 12.4)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download Model Weights

Two model files are required for inference:

| File | Size | Path |
|---|---|---|
| `the_writer.pth.tar` | ~1.8 GB | `demo/checkpoint/` |
| `FaRL-Base-Patch16-LAIONFace20M-ep64.pth` | ~650 MB | `Moeface_project/pretrain/` |

```bash
# Create directories
mkdir -p demo/checkpoint Moeface_project/pretrain

# Download (replace with actual download links)
# wget <download_url> -O demo/checkpoint/the_writer.pth.tar
# wget <download_url> -O Moeface_project/pretrain/FaRL-Base-Patch16-LAIONFace20M-ep64.pth
```

> See the [Releases](../../releases) page for download links.

### 3. Launch Web Demo

```bash
cd demo
python app.py
```

The browser will open `http://localhost:7860` automatically. Three input modes are supported:

- **Image Upload** — Drag and drop or click to upload
- **Webcam Snapshot** — Take a photo and analyze
- **Real-time Video** — Live camera stream analysis (~3 FPS)

### 4. CLI Inference

```bash
cd demo

# Single image inference
python scripts/infer.py --infer_config infer.yaml

# Batch inference (processes all images in image/)
python scripts/batch_infer.py

# Direct inference (no face detection, resize whole image)
python scripts/infer_direct.py --infer_config infer.yaml
```

## Technical Details

### Inference Pipeline

```
Input Image → InsightFace Detection → Dual-resolution Cropping
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                     112×112 crop                       512×512 crop
                          │                                 │
                ┌─────────┼─────────┐             ┌─────────┼─────────┐
                ▼         ▼         ▼             ▼         ▼         ▼
             Recognition  Age   Expression    Parsing  Landmark
              Attribute
```

### Supported Tasks & Labels

**Expression (7 classes):** neutral, happy, sad, surprise, fear, disgust, anger

**Attribute (40 items):** 40 binary CelebA attributes including gender, hair color, eyeglasses, smiling, bangs, etc.

**Face Parsing (19 classes):** background, neck, face, cloth, eyebrows, eyes, iris, nose, lips, hair, glasses, hat, earring, necklace

**Landmarks:** 98 facial landmarks (WFLW dataset)

## Dependencies

| Package | Purpose |
|---|---|
| PyTorch >= 2.6 | Deep learning framework |
| Gradio >= 4.0 | Web demo interface |
| InsightFace | Face detection |
| timm | ViT model components |
| OpenCV | Image processing |
| easydict | Configuration management |

See `requirements.txt` for the full list.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [FaRL](https://github.com/FacePerceiver/FaRL) — Facial representation learning via pre-training
- [InsightFace](https://github.com/deepinsight/insightface) — Face detection and analysis toolkit
- [Faceptor](https://github.com/lxq1000/Faceptor.git) — Multi-task face perception model
