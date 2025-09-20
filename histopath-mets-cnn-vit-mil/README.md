# Histopathology Metastasis Detection: CNN vs ViT vs MIL

A production-ready framework for detecting lymph node metastases in histopathology images using three complementary approaches: Convolutional Neural Networks (CNN), Vision Transformers (ViT), and Multiple Instance Learning (MIL) with attention mechanisms.

## 🔬 Project Overview

This repository implements and compares three state-of-the-art approaches for automated detection of lymph node metastases on the CAMELYON16/17 datasets:

1. **Tile-level CNN**: ResNet-50 backbone with slide-level aggregation
2. **Vision Transformer (ViT)**: ViT-Base/16 for tile classification with attention
3. **Attention-based MIL**: Gated attention mechanism learning from tile embeddings

### Key Features

- 🔄 **End-to-end pipeline**: From whole-slide images to interpretable predictions
- 🧠 **Multiple architectures**: Compare CNN, ViT, and MIL approaches
- 🎯 **Interpretability**: Grad-CAM, attention heatmaps, and SHAP analysis
- 📊 **Robust evaluation**: ROC-AUC, calibration, bootstrap confidence intervals
- 🚀 **Production ready**: MLflow tracking, FastAPI serving, Streamlit app
- ⚡ **Reproducible**: Deterministic training with comprehensive configuration

## 📚 Citations

**Datasets:**
- Bejnordi, B.E., et al. "Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer." JAMA 2017.
- CAMELYON16: https://camelyon16.grand-challenge.org/
- CAMELYON17: https://camelyon17.grand-challenge.org/

**Methods:**
- Ilse, M., et al. "Attention-based Deep Multiple Instance Learning." ICML 2018.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.

## 🛠️ Environment Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- OpenSlide library for WSI processing
- 50+ GB storage for CAMELYON data

### Installation

```bash
# 1. Clone repository
git clone https://github.com/your-org/histopath-mets-cnn-vit-mil.git
cd histopath-mets-cnn-vit-mil

# 2. Create environment
make env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install pre-commit hooks
pre-commit install
```

### Alternative: Docker Setup

```bash
# Build container
docker build -t histopath-mil .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace -p 8501:8501 -p 8000:8000 histopath-mil
```

## 📥 Data Setup

**⚠️ Important**: This repository does not auto-download the CAMELYON datasets due to their size (>100GB) and licensing requirements. You must manually agree to terms and download.

### Step 1: Download CAMELYON16

```bash
# Get download instructions and verify checksums
python scripts/download_camelyon16.py --out data/raw

# Follow the printed instructions to download:
# - Training slides and masks
# - Test slides  
# - Label files
```

### Step 2: Prepare Dataset

```bash
# Process metadata and create patient splits
python scripts/prepare_camelyon16.py --in data/raw --out data/processed
```

### Step 3: Generate Tiles

```bash
# Extract tissue tiles from whole-slide images
python scripts/tile_wsi.py \
    --in data/processed \
    --out data/tiles \
    --tile-size 256 \
    --magnification 10 \
    --otsu-mask
```

### Step 4: Build MIL Bags

```bash
# Group tiles into bags for MIL training
python scripts/build_mil_bags.py \
    --tiles data/tiles \
    --out data/bags \
    --min-tiles 50
```

## 🚀 Training Models

All training uses Hydra configuration and MLflow tracking:

### CNN Baseline
```bash
python -m src.histopath.train.train model=cnn +exp=baseline_cnn
```

### Vision Transformer
```bash
python -m src.histopath.train.train model=vit +exp=baseline_vit
```

### MIL with Attention
```bash
python -m src.histopath.train.train model=mil +exp=attention_mil
```

### Custom Configuration
```bash
# Override specific parameters
python -m src.histopath.train.train \
    model=mil \
    model.hidden_dim=512 \
    training.lr=1e-4 \
    training.batch_size=32 \
    +exp=custom_mil
```

## 📊 Evaluation & Analysis

### Model Evaluation
```bash
# Compute comprehensive metrics
python -m src.histopath.eval.evaluate \
    ckpt=path/to/checkpoint.ckpt \
    eval=test
```

### Generate Interpretability
```bash
# Create Grad-CAM and attention heatmaps  
python -m src.histopath.eval.interpretability \
    ckpt=path/to/checkpoint.ckpt \
    slide_id=test_001 \
    --method gradcam
```

### WSI Heatmap Generation
```bash
# Generate full slide attention heatmaps
python scripts/infer_wsi_heatmap.py \
    --slide data/processed/test/test_001.tif \
    --ckpt checkpoints/mil_best.ckpt \
    --out viz/heatmaps
```

## 🎯 Visualization & Serving

### Interactive Streamlit App
```bash
# Launch visualization dashboard
streamlit run src/histopath/app/streamlit_app.py

# Navigate to http://localhost:8501
```

### FastAPI Serving
```bash
# Start prediction API
uvicorn src.histopath.serve.api:app --reload

# API docs at http://localhost:8000/docs
```

### Example API Usage
```python
import requests

# Score a bag of tiles
response = requests.post(
    "http://localhost:8000/score_bag",
    json={
        "tiles": ["tile1.jpg", "tile2.jpg"],
        "model_type": "mil"
    }
)
prediction = response.json()
```

## 📈 MLflow Tracking

All experiments are automatically tracked with MLflow:

```bash
# View experiment dashboard
mlflow ui

# Navigate to http://localhost:5000
```

**Tracked metrics:**
- Training/validation loss and AUC
- Test set performance metrics
- Model hyperparameters
- Attention visualizations
- Calibration curves

## 🔬 Interpretability Features

### Grad-CAM (CNN/ViT)
- Tile-level activation heatmaps
- Class activation mapping
- Aggregated slide-level importance

### Attention Heatmaps (MIL)
- Per-tile attention scores
- WSI-level attention overlays
- Interactive attention exploration

### SHAP Analysis
- Feature importance for embedding layers
- Kernel SHAP for model explanations
- DeepSHAP for gradient-based attribution

## 📊 Expected Performance

| Model | ROC-AUC | PR-AUC | Sens@90%Spec | Notes |
|-------|---------|--------|--------------|-------|
| CNN Baseline | 0.85-0.90 | 0.70-0.80 | 0.65-0.75 | Fast inference |
| ViT | 0.87-0.92 | 0.72-0.82 | 0.68-0.78 | Better feature learning |
| MIL | 0.90-0.95 | 0.75-0.85 | 0.75-0.85 | Interpretable attention |

*Performance varies by train/test split and data preprocessing choices*

## 🔄 Reproducibility

All experiments are fully reproducible:

- **Deterministic training**: Global seeds, CUDNN deterministic
- **Pinned dependencies**: Exact version specifications
- **Config snapshots**: Saved with each MLflow run
- **Patient-level splits**: Prevent data leakage

### K-Fold Cross-Validation
```bash
# Run 5-fold CV with patient stratification
python -m src.histopath.train.train \
    model=mil \
    training.cv_folds=5 \
    training.stratify_patients=true
```

## 🧪 Development & Testing

### Running Tests
```bash
# Run full test suite
make test

# Run specific test modules
pytest tests/test_mil_attention.py -v

# Generate coverage report
pytest --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
make format

# Run linters
make lint

# Type checking
make typecheck
```

## 🚨 Ethical Considerations

**Data Privacy:**
- CAMELYON datasets are anonymized research datasets
- No patient identifiers are stored or transmitted
- Follow institutional guidelines for medical AI research

**Clinical Usage:**
- This is a research implementation, not FDA-approved
- Requires clinical validation before any diagnostic use
- Human oversight essential for medical decision making

**Bias & Fairness:**
- Models may exhibit bias from training data distribution
- Evaluate performance across different patient populations
- Consider domain adaptation for different hospitals/scanners

## 📝 Configuration Guide

### Hydra Configuration Structure

```yaml
# configs/config.yaml (main config)
defaults:
  - data: camelyon16
  - model: mil
  - training: default
  - paths: local

# Override any parameter:
python -m src.histopath.train.train \
    model.attention.hidden_dim=256 \
    training.lr=1e-3
```

### Key Configuration Files
- `configs/data.yaml`: Dataset paths and preprocessing
- `configs/train_*.yaml`: Model-specific training settings
- `configs/eval.yaml`: Evaluation configuration
- `configs/paths.yaml`: Data and output paths

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run development checks
make check-all
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CAMELYON challenge organizers for the dataset
- PyTorch Lightning team for the training framework
- OpenSlide contributors for WSI processing tools
- The broader computational pathology research community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/histopath-mets-cnn-vit-mil/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/histopath-mets-cnn-vit-mil/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-org/histopath-mets-cnn-vit-mil/wiki)

---

**⭐ If this project helps your research, please consider citing and starring the repository!**