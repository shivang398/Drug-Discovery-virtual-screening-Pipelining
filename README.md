# Molecular Property Prediction Pipeline

## Overview

This project provides a comprehensive machine learning pipeline for drug discovery, focusing on molecular property prediction using advanced machine learning techniques. The pipeline leverages RDKit for molecular processing, TensorFlow for deep learning, and scikit-learn for data preprocessing and evaluation.

## Features

### Key Capabilities
- Molecular fingerprint generation
- Advanced data augmentation for small datasets
- Scaffold-based train/test splitting
- Anti-overfitting strategies
- Flexible model architecture
- Comprehensive model evaluation and visualization

### Machine Learning Techniques
- Morgan Fingerprint calculation
- Data augmentation via molecular transformations
- Regularized neural network models
- Early stopping and adaptive learning rate
- Feature scaling
- Cyclic learning rate scheduling

## Prerequisites

### Required Libraries
- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- rdkit
- google.colab (optional)

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow rdkit
```

## Project Structure

```
drug-discovery-pipeline/
│
├── bace.csv                 # Sample dataset
├── drug_discovery_pipeline.py  # Main pipeline script
└── README.md                # Project documentation
```

## Quick Start

### Basic Usage

```python
from drug_discovery_pipeline import DrugDiscoveryPipeline

# Initialize pipeline
pipeline = DrugDiscoveryPipeline(
    dataset_name="bace_classification",
    split_type="scaffold"
)

# Load data
pipeline.load_data(
    compound_file="bace.csv",
    smiles_column='mol',
    label_column='Class'
)

# Augment data
pipeline.augment_data_method()

# Scale features
pipeline.scale_features()

# Build and train model
pipeline.build_model(
    model_type='classification',
    hidden_units=(64, 32),
    dropout=0.4
)

pipeline.train()

# Evaluate model
pipeline.evaluate(dataset='test')

# Visualize performance
pipeline.visualize_model_performance()
```

## Advanced Features

### Data Augmentation
The pipeline supports molecular data augmentation through various transformations:
- Canonical SMILES generation
- Tautomer enumeration
- 3D conformer generation
- Stereochemistry modifications

### Model Architecture
- Flexible neural network architecture
- Customizable layer units
- Dropout regularization
- L1/L2 regularization
- Batch normalization

### Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1-Score, ROC AUC
- Regression: MSE, MAE, R² Score

## Prediction Function

```python
new_compounds = [
    "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
    # Add more SMILES strings
]

predictions = predict_new_compounds(pipeline, new_compounds)
```

## Visualization Capabilities

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Actual vs Predicted Plot
- Residual Analysis
- Prediction Distribution

## Customization Options

- Scaffold-based or random data splitting
- Custom fingerprint generation
- Adjustable model hyperparameters
- Optional feature reduction via PCA

## Google Colab Support

The pipeline includes built-in support for Google Colab, with methods to:
- Detect Colab environment
- Mount Google Drive
- Load datasets from Drive

## Limitations

- Requires valid SMILES representations
- Performance depends on dataset quality
- Computational resources for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
