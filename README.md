# Drug Discovery Pipeline

This repository contains a Python-based pipeline for drug discovery and molecular data analysis. It includes tools for molecular descriptor calculation, data augmentation, and machine learning-based drug activity prediction.

## Features

- **Molecular Descriptor Calculation**: Generate molecular fingerprints (Morgan fingerprints) for molecules using RDKit.
- **Data Augmentation**: Create synthetic molecular variants to expand datasets.
- **Drug Discovery Pipeline**: A comprehensive pipeline for loading datasets, splitting data, and training machine learning models with anti-overfitting measures.
- **Active Learning**: Optional active learning for iterative model improvement.
- **Support for CSV and SDF Files**: Load compound data from various file formats.

## Requirements

The following Python libraries are required to run the code:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `rdkit`
- `scikit-learn`
- `tensorflow`

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow rdkit-pypi
```

## File Structure

- **`MolecularDescriptorCalculator`**: A class to compute Morgan fingerprints for molecules.
- **`MolecularDataAugmentation`**: A class to perform data augmentation by generating synthetic molecular variants.
- **`DrugDiscoveryPipeline`**: The main pipeline for dataset loading, splitting, and training models.

## Usage

### 1. Check Environment (Google Colab or Local)
The script includes utility functions to check if it is running in Google Colab and mount Google Drive if needed.

```python
check_if_colab()
mount_google_drive()
```

### 2. Load Data
Load molecular data from a CSV or SDF file. The dataset should include SMILES strings and corresponding labels.

```python
pipeline = DrugDiscoveryPipeline(dataset_name="bace_classification")
pipeline.load_data(compound_file="path_to_file.csv", smiles_column="mol", label_column="Class")
```

### 3. Data Augmentation
Generate synthetic molecules to expand the dataset.

```python
augmented_mols, augmented_labels = MolecularDataAugmentation.generate_synthetic_molecules(
    mols=original_mols,
    labels=original_labels,
    augmentation_factor=2
)
```

### 4. Train/Test Split
The pipeline supports both random and scaffold-based splits with a default ratio of 70/30.

```python
pipeline.load_data(test_size=0.3)
```

### 5. Active Learning (Optional)
Enable active learning to iteratively improve the model by selecting informative samples.

```python
pipeline = DrugDiscoveryPipeline(active_learning=True)
```

## Example Dataset

The example dataset used in this script is named `bace_classification`. It includes SMILES strings representing molecules and their binary activity labels (`Active` or `Inactive`).

### CSV File Format

| mol         | Class     |
|-------------|-----------|
| CC(=O)OC1=... | Active    |
| CC(=O)NC1C... | Inactive  |

### SDF File Format

For SDF files, ensure that the activity information is stored as a property (`activity`) in each molecule.

## Key Functions

### MolecularDescriptorCalculator

Calculate Morgan fingerprints for molecules:

```python
calculator = MolecularDescriptorCalculator(fp_radius=2, fp_bits=1024)
fingerprints = calculator.calculate_for_mols(mols)
```

### MolecularDataAugmentation

Perform transformations to generate synthetic molecules:

```python
variants = MolecularDataAugmentation.apply_transformations(molecule, num_variants=5)
```

### DrugDiscoveryPipeline

Load data, split datasets, and analyze:

```python
pipeline.load_data(compound_file="data.csv", smiles_column="mol", label_column="Class")
```

## Notes

1. Ensure that RDKit is properly installed for molecular operations.
2. The script provides warnings if invalid SMILES strings are encountered during processing.
3. Labels should be binary (`0` or `1`). Non-binary labels will be converted or raise an error.
