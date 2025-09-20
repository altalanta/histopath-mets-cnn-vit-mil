.PHONY: help env install lint format typecheck test clean app serve train-cnn train-vit train-mil check-all

help:
	@echo "Available targets:"
	@echo "  env          - Create conda environment"
	@echo "  install      - Install dependencies"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black, isort)"
	@echo "  typecheck    - Run type checking (mypy)"
	@echo "  test         - Run test suite"
	@echo "  clean        - Clean cache and temp files"
	@echo "  app          - Launch Streamlit app"
	@echo "  serve        - Start FastAPI server"
	@echo "  train-cnn    - Train CNN baseline"
	@echo "  train-vit    - Train ViT model"
	@echo "  train-mil    - Train MIL model"
	@echo "  check-all    - Run all checks (lint, format, typecheck, test)"

env:
	conda env create -f environment.yml

install:
	pip install -r requirements.txt
	pip install -e .

lint:
	flake8 src/ scripts/ tests/
	
format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

app:
	streamlit run src/histopath/app/streamlit_app.py

serve:
	uvicorn src.histopath.serve.api:app --reload --host 0.0.0.0 --port 8000

train-cnn:
	python -m src.histopath.train.train model=cnn +exp=baseline_cnn

train-vit:
	python -m src.histopath.train.train model=vit +exp=baseline_vit

train-mil:
	python -m src.histopath.train.train model=mil +exp=attention_mil

check-all: lint typecheck test
	@echo "All checks passed!"

# Data preparation targets
download-data:
	python scripts/download_camelyon16.py --out data/raw

prepare-data:
	python scripts/prepare_camelyon16.py --in data/raw --out data/processed

tile-data:
	python scripts/tile_wsi.py --in data/processed --out data/tiles --tile-size 256 --magnification 10 --otsu-mask

build-bags:
	python scripts/build_mil_bags.py --tiles data/tiles --out data/bags --min-tiles 50

# Full pipeline
setup-data: download-data prepare-data tile-data build-bags

# Docker targets
docker-build:
	docker build -t histopath-mil .

docker-run:
	docker run --gpus all -v $(PWD):/workspace -p 8501:8501 -p 8000:8000 histopath-mil

# Development
dev-install:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

# MLflow
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000