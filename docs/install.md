# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip >= 22
- For CUDA workflows, an NVIDIA GPU with drivers >= 525

## CPU install

```bash
pip install histo-omics-lite
```

Optional extras:

```bash
pip install histo-omics-lite[viz]        # plotting stack
pip install histo-omics-lite[dev,viz]    # development and docs
```

## CUDA support

Install the CUDA extra on top of the official PyTorch wheels:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install histo-omics-lite[cuda,viz]
```

> **Note**: The `cuda` extra installs monitoring helpers (`cuda-python`, `pynvml`). The correct CUDA-enabled PyTorch wheel still needs to be selected manually via the extra index URL above.

## Editable install for development

```bash
git clone https://github.com/altalanta/histo-omics-lite.git
cd histo-omics-lite
pip install -e .[dev,viz]
pre-commit install
```
