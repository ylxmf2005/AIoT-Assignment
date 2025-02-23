# AIoT Assignment - Cat-Dog Classification

## Authors
- Ethan Lee
- Tommy Cheung
- Koon Chris

## Introduction

This project implements and compares different approaches for cat-dog image classification:
- Support Vector Machine (SVM)
- K-Means Clustering
- Convolutional Neural Network (CNN)
- Spiking Neural Network (SNN)
- Autoencoders

## Requirements

- Python 3.10+
- CUDA 11.8 (for GPU support)
- `uv` package manager

> **Note**: If you encounter any dependency issues, you may need to modify `pyproject.toml` to match your system configuration. Refer to the [uv documentation](https://docs.astral.sh/uv/) for troubleshooting.

## Setup and Installation

1. Install uv:
```bash
# Using curl (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc # or source ~/.zshrc, depending on your shell

# Using PowerShell (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Install dependencies:
```bash
uv sync 
```

## Experiment Tracking with Weights & Biases

Before running the models, set up wandb:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Login to wandb:
```bash
wandb login
```
3. View results:
   - Real-time training metrics: https://wandb.ai/your-username/project-name
   - Local results in `wandb/` directory
   - Detailed metrics in `wandb/run-{timestamp}/files/metrics.json` for each run

## Running Models

Choose a model to run:
```bash
# Run CNN model
uv run python cnn.py

# Run SNN model
uv run python snn.py

# Run SVM model
uv run python svm.py

# Run KMeans model
uv run python kmeans.py
```
