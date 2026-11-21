# Deep Learning Approaches for Reliable Screening of Autism Spectrum Disorder

## Overview

This thesis project studies automated Autism Spectrum Disorder (ASD) screening using EEG and MRI data. It includes preprocessing, feature extraction, deep-learning model training, multimodal analysis, uncertainty estimation, performance evaluation, and result visualization.

## Thesis

The complete thesis is available in PDF format:

- [`Deep-Learning-Approaches-for-Reliable-ASD-Screening.pdf`](Deep-Learning-Approaches-for-Reliable-ASD-Screening.pdf)

## Repository Structure

- `src/` — preprocessing, model, training, EEG, MRI, and multimodal-analysis programs
- `notebooks/` — exploratory data analysis
- `data/` — source EEG recordings and supporting metadata
- `processed/` — generated processed artifacts
- `models/` — trained model files
- `results/` — figures and evaluation outputs
- `mri/` — MRI data and documentation
- `webapp/` — demonstration interface
- `requirements.txt` — Python dependencies
- `references.bib` — literature references

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main Workflows

```bash
python src/preprocess.py
python src/create_spectrograms.py
python src/train.py
python src/evaluate.py
python src/multimodal_fusion.py
```

The notebooks directory contains the exploratory workflow used during the project.

## Research Scope

This project is an academic research tool. It does not provide a medical diagnosis and should not replace assessment by a qualified healthcare professional.
