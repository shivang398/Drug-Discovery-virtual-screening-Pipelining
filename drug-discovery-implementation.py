import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import deepchem as dc
from deepchem.models import GCNModel  # Correct import for DeepChem 2.8.0
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.splits import RandomSplitter, ScaffoldSplitter
from deepchem.data import NumpyDataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras


class DrugDiscoveryPipeline:
    """
    End-to-end pipeline for drug discovery using DeepChem.
    """

    def __init__(self, dataset_name="bace_classification", featurizer_type="graph_conv",
                 split_type="scaffold", active_learning=True):
        """
        Initialize the drug discovery pipeline.

        Parameters
        ----------
        dataset_name : str
            Name of the MoleculeNet dataset to use
        featurizer_type : str
            Type of featurizer to use ('graph_conv')
        split_type : str
            Type of data split ('random', 'scaffold')
        active_learning : bool
            Whether to use active learning for compound selection
        """
        self.dataset_name = dataset_name
        self.featurizer_type = featurizer_type
        self.split_type = split_type
        self.active_learning = active_learning
        self.model = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.unlabeled_pool = None
        self.active_learning_history = []
        self.tasks = None

    def load_data(self, smiles_list=None, labels=None, compound_file=None):
        """Load and preprocess the dataset."""
        if smiles_list is None and compound_file is None:
            raise ValueError("Provide either smiles_list or compound_file")
        # Load compounds from file
        if compound_file:
            if compound_file.endswith('.csv'):
                df = pd.read_csv(compound_file)
                if 'smiles' in df.columns and 'label' in df.columns:
                    smiles_list = df['smiles'].tolist()
                    labels = df['label'].tolist()
                    # Convert labels to a numpy array of shape (n_samples, 1)
                    labels = np.array(labels)
                    labels = labels.reshape(-1, 1)
                else:
                    raise ValueError("CSV file must contain 'smiles' and 'label' columns")
            elif compound_file.endswith('.sdf'):
                suppl = Chem.SDMolSupplier(compound_file)
                mols = [mol for mol in suppl if mol is not None]
                smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
                # Extract properties or labels from SDF if available
                labels = [int(mol.GetProp('activity')) if mol.HasProp('activity') else 0 for mol in mols]
                labels = np.array(labels)
                labels = labels.reshape(-1, 1)
            else:
                raise ValueError("Unsupported file format. Use CSV or SDF.")
        else:
            if labels is None:
                raise ValueError("Provide label list and smiles list.")
        print(f"Loading custom dataset...")

        # Choose featurizer
        featurizer = MolGraphConvFeaturizer()

        # Featurize the molecules
        features = featurizer.featurize(smiles_list)

        # Convert to numpy arrays if needed
        X = features
        y = np.array(labels)
        ids = np.array(smiles_list)  # Use SMILES as IDs

        # Split data
        if self.split_type == "scaffold" and len(smiles_list) > 10:
            # For scaffold splitting, first convert to molecules
            mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            splitter = ScaffoldSplitter()
            train_idx, valid_idx, test_idx = splitter.split(mols, frac_train=0.7, frac_valid=0.15, frac_test=0.15)

            # Create datasets based on indices
            train_X = [X[i] for i in train_idx]
            valid_X = [X[i] for i in valid_idx]
            test_X = [X[i] for i in test_idx]

            self.train_dataset = NumpyDataset(train_X, y[train_idx], ids=ids[train_idx])
            self.valid_dataset = NumpyDataset(valid_X, y[valid_idx], ids=ids[valid_idx])
            self.test_dataset = NumpyDataset(test_X, y[test_idx], ids=ids[test_idx])
        else:
            # Random split
            indices = list(range(len(smiles_list)))
            train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
            valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

            # Create DeepChem datasets - handle list of features
            train_X = [X[i] for i in train_idx]
            valid_X = [X[i] for i in valid_idx]
            test_X = [X[i] for i in test_idx]

            self.train_dataset = NumpyDataset(train_X, y[train_idx], ids=ids[train_idx])
            self.valid_dataset = NumpyDataset(valid_X, y[valid_idx], ids=ids[valid_idx])
            self.test_dataset = NumpyDataset(test_X, y[test_idx], ids=ids[test_idx])

        self.tasks = ["activity"]

        print(f"Dataset loaded with {len(self.train_dataset)} training samples,",
              f"{len(self.valid_dataset)} validation samples, and {len(self.test_dataset)} test samples.")
        print(f"Tasks: {self.tasks}")

        # Analyze dataset
        self._analyze_dataset()

        return self

    def _analyze_dataset(self):
        """Analyze the dataset and print statistics."""
        # Count positives and negatives in training set
        y_train = self.train_dataset.y
        n_tasks = len(self.tasks)

        print("\nDataset statistics:")
        for i, task in enumerate(self.tasks):
            positives = np.sum(y_train[:, i] == 1)
            negatives = np.sum(y_train[:, i] == 0)
            print(f"Task '{task}': {positives} positives, {negatives} negatives")

    def build_model(self, n_tasks=None, model_dir="./model",
                    batch_size=128, learning_rate=0.001, n_layers=2):
        """
        Build and compile a Graph Convolutional Network model.
        """
        if self.tasks is None:
            raise ValueError("Tasks have not been loaded. Call load_data() first.")

        if n_tasks is None:
            n_tasks = len(self.tasks)

        # Determine the model type (classification or regression)
        model_type = "classification"
        if "regression" in self.dataset_name:
            model_type = "regression"

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Build the model with GCNModel in DeepChem 2.8.0
        print(f"Building GCNModel with {n_tasks} tasks, mode={model_type}")

        # Configure optimizer with Keras 3.0 syntax
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model = GCNModel(
            n_tasks=n_tasks,
            mode=model_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_dir=model_dir,
            predictor_hidden_feats=128,
            num_graph_conv_layers=n_layers,
            graph_conv_dim=64,
            optimizer=optimizer  # Pass optimizer directly
        )

        return self

    def train_model(self, epochs=50, patience=10):
        """
        Train the model with early stopping.

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train for
        patience : int
            Number of epochs to wait for improvement before stopping
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        print(f"Training model for up to {epochs} epochs with patience={patience}...")

        # Define metric
        if self.model.mode == "classification":
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        else:
            metric = dc.metrics.Metric(dc.metrics.r2_score)

        # Train with early stopping
        train_scores, valid_scores = [], []
        best_score = -np.inf
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            # Train for one epoch
            loss = self.model.fit(self.train_dataset, nb_epoch=1)

            # Evaluate
            train_score = self.model.evaluate(self.train_dataset, [metric])
            valid_score = self.model.evaluate(self.valid_dataset, [metric])

            # Print progress
            mean_train = np.mean(list(train_score.values()))
            mean_valid = np.mean(list(valid_score.values()))
            train_scores.append(mean_train)
            valid_scores.append(mean_valid)

            print(f"Epoch {epoch + 1}/{epochs}: train = {mean_train:.4f}, valid = {mean_valid:.4f}")

            # Check for improvement
            if mean_valid > best_score:
                best_score = mean_valid
                best_epoch = epoch
                patience_counter = 0
                # Save the best model
                self.model.save_checkpoint(model_dir=self.model.model_dir)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch + 1}.")
                    break

        # Restore best model
        self.model.restore()

        # Evaluate on test set
        test_score = self.model.evaluate(self.test_dataset, [metric])
        print(f"Test score: {np.mean(list(test_score.values())):.4f}")

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label='Train')
        plt.plot(valid_scores, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model.model_dir, 'training_history.png'))

        return self

    def setup_active_learning(self, external_compounds_file, initial_size=100, batch_size=10):
        """
        Set up the active learning pipeline with an external library of compounds.

        Parameters
        ----------
        external_compounds_file : str
            Path to a CSV or SDF file containing external compounds to screen
        initial_size : int
            Number of compounds to include in the initial training set
        batch_size : int
            Number of compounds to select in each active learning iteration
        """
        if not self.active_learning:
            print("Active learning is disabled. Skipping setup.")
            return self

        print(f"Setting up active learning pipeline with {external_compounds_file}...")

        # Load external compounds
        if external_compounds_file.endswith('.csv'):
            df = pd.read_csv(external_compounds_file)
            if 'smiles' in df.columns:
                smiles_list = df['smiles'].tolist()
            else:
                raise ValueError("CSV file must contain a 'smiles' column")
        elif external_compounds_file.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(external_compounds_file)
            smiles_list = [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
        else:
            raise ValueError("Unsupported file format. Use CSV or SDF.")

        print(f"Loaded {len(smiles_list)} external compounds for screening.")

        # Featurize the compounds
        featurizer = MolGraphConvFeaturizer()
        X = featurizer.featurize(smiles_list)
        y = np.zeros((len(X), len(self.tasks)))  # Initialize with placeholder labels
        ids = np.array(smiles_list)

        # Create unlabeled pool
        self.unlabeled_pool = NumpyDataset(X, y, ids=ids)
        print(f"Created unlabeled pool with {len(self.unlabeled_pool)} compounds.")

        # Create initial active learning set - ensuring we don't exceed the dataset size
        initial_size = min(initial_size, len(self.train_dataset))

        # Extract the initial training subset
        initial_X = [self.train_dataset.X[i] for i in range(initial_size)]
        initial_y = self.train_dataset.y[:initial_size]
        initial_ids = self.train_dataset.ids[:initial_size]

        self.active_learning_set = NumpyDataset(initial_X, initial_y, ids=initial_ids)

        self.al_batch_size = batch_size
        print(f"Active learning initialized with {initial_size} compounds and batch size {batch_size}.")

        return self
    def run_active_learning_cycle(self, n_cycles=5, oracle_function=None):
        """
        Run active learning cycles to iteratively improve the model.

        Parameters
        ----------
        n_cycles : int
            Number of active learning cycles to run
        oracle_function : callable
            Function that takes SMILES strings and returns labels (simulates experimental validation)
            If None, we use uncertainty sampling without updating the model
        """
        if not self.active_learning:
            print("Active learning is disabled. Skipping.")
            return self

        if self.unlabeled_pool is None:
            raise ValueError("Active learning has not been set up. Call setup_active_learning() first.")

        print(f"Starting active learning cycle with {n_cycles} iterations...")

        for cycle in range(n_cycles):
            print(f"\nActive Learning Cycle {cycle+1}/{n_cycles}")

            # Rebuild and train the model on the current active learning set
            model_dir = os.path.join(self.model.model_dir, f"al_cycle_{cycle}")
            os.makedirs(model_dir, exist_ok=True)

            # Create a new model instance
            self.build_model(model_dir=model_dir)

            # Train on the active learning set
            print(f"Training model on active learning set ({len(self.active_learning_set)} compounds)...")
            self.model.fit(self.active_learning_set, nb_epoch=50)

            # Evaluate the model
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score if self.model.mode == "classification"
                                    else dc.metrics.r2_score)
            test_score = self.model.evaluate(self.test_dataset, [metric])
            print(f"Test score: {np.mean(list(test_score.values())):.4f}")

            # Select compounds for the next batch
            selected_indices = self._select_compounds(batch_size=self.al_batch_size)
            selected_smiles = [self.unlabeled_pool.ids[i] for i in selected_indices]

            print(f"Selected {len(selected_smiles)} compounds for labeling.")

            if oracle_function is not None:
                # Get labels from oracle function
                new_labels = oracle_function(selected_smiles)

                # Create a new dataset with the selected compounds
                selected_X = [self.unlabeled_pool.X[i] for i in selected_indices]
                selected_y = np.array(new_labels).reshape(-1, len(self.tasks))
                selected_ids = np.array(selected_smiles)

                new_data = NumpyDataset(selected_X, selected_y, ids=selected_ids)

                # Merge with existing active learning set
                self.active_learning_set = dc.data.NumpyDataset.merge([self.active_learning_set, new_data])

                # Remove selected compounds from unlabeled pool
                mask = np.ones(len(self.unlabeled_pool), dtype=bool)
                mask[selected_indices] = False

                X_remaining = [self.unlabeled_pool.X[i] for i in range(len(self.unlabeled_pool)) if mask[i]]
                y_remaining = self.unlabeled_pool.y[mask]
                ids_remaining = self.unlabeled_pool.ids[mask]

                self.unlabeled_pool = NumpyDataset(X_remaining, y_remaining, ids=ids_remaining)

                # Track the history
                self.active_learning_history.append({
                    'cycle': cycle+1,
                    'test_score': np.mean(list(test_score.values())),
                    'selected_compounds': selected_smiles,
                    'pool_size': len(self.unlabeled_pool)
                })

                print(f"Updated active learning set to {len(self.active_learning_set)} compounds.")
                print(f"Unlabeled pool now contains {len(self.unlabeled_pool)} compounds.")
            else:
                print("No oracle function provided. Selected compounds:")
                for smiles in selected_smiles:
                    print(f"- {smiles}")
                print("Active learning simulation without updating the model.")

        # Plot active learning history
        if oracle_function is not None and self.active_learning_history:
            self._plot_active_learning_history()

        return self
    def _select_compounds(self, batch_size=10, strategy="uncertainty"):
        """
        Select compounds from the unlabeled pool using various strategies.

        Parameters
        ----------
        batch_size : int
            Number of compounds to select
        strategy : str
            Selection strategy ('uncertainty', 'diversity', or 'random')

        Returns
        -------
        list
            Indices of selected compounds
        """
        if strategy == "random":
            return np.random.choice(len(self.unlabeled_pool), batch_size, replace=False)

        # Get predictions and uncertainties for all compounds in the pool
        y_pred, uncertainty = self._get_uncertainty(self.unlabeled_pool)

        if strategy == "uncertainty":
            # Select compounds with highest uncertainty
            selected_indices = np.argsort(-uncertainty)[:batch_size]
        elif strategy == "diversity":
            # Implement diversity-based selection (e.g., k-means clustering)
            # This is a simplified version using only uncertainty
            selected_indices = []

            # First, select the compound with highest uncertainty
            selected_indices.append(np.argmax(uncertainty))

            # Then select the rest based on a combination of uncertainty and diversity
            remaining = list(range(len(self.unlabeled_pool)))
            remaining.remove(selected_indices[0])

            while len(selected_indices) < batch_size and remaining:
                best_idx = -1
                best_score = -np.inf

                for idx in remaining:
                    # Score is a combination of uncertainty and diversity from already selected
                    score = uncertainty[idx]
                    best_idx = idx if score > best_score else best_idx
                    best_score = max(score, best_score)

                if best_idx >= 0:
                    selected_indices.append(best_idx)
                    remaining.remove(best_idx)
                else:
                    break
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

        return selected_indices
    def _get_uncertainty(self, dataset):
        """
        Get prediction uncertainty for all compounds in a dataset.

        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset to get uncertainty for

        Returns
        -------
        tuple
            (predictions, uncertainty)
        """
        y_pred = self.model.predict(dataset)

        if self.model.mode == "classification":
            # For binary classification, uncertainty is highest at p=0.5
            uncertainty = -np.abs(y_pred - 0.5)
        else:
            # For regression, we don't have a good uncertainty measure
            # without using ensemble or dropout. Use a random value for now.
            uncertainty = np.random.random(len(dataset))
        
        return y_pred, uncertainty
    def _plot_active_learning_history(self):
        """Plot the active learning history."""
        if not self.active_learning_history:
            print("No active learning history to plot.")
            return

        cycles = [h['cycle'] for h in self.active_learning_history]
        scores = [h['test_score'] for h in self.active_learning_history]
        pool_sizes = [h['pool_size'] for h in self.active_learning_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot test scores
        ax1.plot(cycles, scores, 'o-', markersize=8)
        ax1.set_xlabel('Active Learning Cycle')
        ax1.set_ylabel('Test Score')
        ax1.set_title('Performance vs. Active Learning Cycle')
        ax1.grid(True)

        # Plot pool size
        ax2.plot(cycles, pool_sizes, 's-', markersize=8, color='green')
        ax2.set_xlabel('Active Learning Cycle')
        ax2.set_ylabel('Unlabeled Pool Size')
        ax2.set_title('Unlabeled Pool Size vs. Cycle')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model.model_dir, 'active_learning_history.png'))
        plt.close()
    def predict(self, smiles_list):
        """
        Make predictions for a list of SMILES strings.

        Parameters
        ----------
        smiles_list : list
            List of SMILES strings to make predictions for

        Returns
        -------
        numpy.ndarray
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        # Featurize molecules
        featurizer = MolGraphConvFeaturizer()
        features = featurizer.featurize(smiles_list)

        # Create dataset
        X = features
        y = np.zeros((len(X), len(self.tasks)))  # Placeholder labels
        ids = np.array(smiles_list)
        dataset = NumpyDataset(X, y, ids=ids)

        # Make predictions
        return self.model.predict(dataset)
       
    def visualize_predictions(self, smiles_list, labels=None, n_per_row=4, conf_threshold=0.5):
        """
        Visualize molecules with their predictions.

        Parameters
        ----------
        smiles_list : list
            List of SMILES strings to visualize
        labels : list, optional
            True labels for the molecules, if available
        n_per_row : int
            Number of molecules per row in the visualization
        conf_threshold : float
            Confidence threshold for coloring predictions
        """
        # Convert SMILES to RDKit molecules
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

        # Make predictions
        preds = self.predict(smiles_list)

        # Prepare visualization
        n_mols = len(mols)
        n_rows = (n_mols + n_per_row - 1) // n_per_row

        # Add prediction information to the molecules
        legends = []
        for i, (mol, pred) in enumerate(zip(mols, preds)):
            prediction = pred[0]  # Assuming single task

            if self.model.mode == "classification":
                label = f"Pred: {prediction:.2f}"
                if labels is not None:
                    label += f", True: {labels[i]}"
                legends.append(label)

                # Color atoms based on prediction
                if prediction >= conf_threshold:
                    color = (0, 1, 0)  # Green for active (high confidence)
                elif prediction <= 1 - conf_threshold:
                    color = (1, 0, 0)  # Red for inactive (high confidence)
                else:
                    color = (0.7, 0.7, 0)  # Yellow for uncertain
            else:
                # For regression
                label = f"Pred: {prediction:.2f}"
                if labels is not None:
                    label += f", True: {labels[i]:.2f}"
                legends.append(label)

        # Generate image
        img = Draw.MolsToGridImage(mols, molsPerRow=n_per_row, subImgSize=(200, 200),
                                  legends=legends)

        # Save and display image
        img_file = os.path.join(self.model.model_dir, 'predicted_molecules.png')
        img.save(img_file)

        return img
    def evaluate_model(self, dataset=None, metrics=None):
        """
        Evaluate the model on a dataset.

        Parameters
        ----------
        dataset : dc.data.Dataset, optional
            Dataset to evaluate on. If None, uses the test dataset.
        metrics : list, optional
            List of metrics to evaluate. If None, uses default metrics.

        Returns
        -------
        dict
            Dictionary of metric results
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        if dataset is None:
            dataset = self.test_dataset

        if metrics is None:
            if self.model.mode == "classification":
                metrics = [
                    dc.metrics.Metric(dc.metrics.roc_auc_score),
                    dc.metrics.Metric(dc.metrics.prc_auc_score),
                    dc.metrics.Metric(dc.metrics.accuracy_score),
                    dc.metrics.Metric(dc.metrics.recall_score),
                    dc.metrics.Metric(dc.metrics.precision_score)
                ]
            else:
                metrics = [
                    dc.metrics.Metric(dc.metrics.r2_score),
                    dc.metrics.Metric(dc.metrics.mean_squared_error),
                    dc.metrics.Metric(dc.metrics.mean_absolute_error)
                ]

        results = self.model.evaluate(dataset, metrics)

        print("\nModel Evaluation Results:")
        for name, value in results.items():
            print(f"{name}: {value:.4f}")

        return results

    def plot_roc_curve(self):
        """Plot the ROC curve for the model."""
        if self.model.mode != "classification":
            print("ROC curve is only available for classification models.")
            return

        # Get predictions and true labels
        y_true = self.test_dataset.y
        y_pred = self.model.predict(self.test_dataset)

        # Compute ROC curve and ROC area for each task
        plt.figure(figsize=(10, 8))

        for i, task in enumerate(self.tasks):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{task} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.savefig(os.path.join(self.model.model_dir, 'roc_curve.png'))
        plt.close()
def plot_precision_recall_curve(self):
    """Plot the Precision-Recall curve for the model."""
    if self.model.mode != "classification":
        print("Precision-Recall curve is only available for classification models.")
        return

    # Get predictions and true labels
    y_true = self.test_dataset.y
    y_pred = self.model.predict(self.test_dataset)

    # Compute Precision-Recall curve and area for each task
    plt.figure(figsize=(10, 8))

    for i, task in enumerate(self.tasks):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f'{task} (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.savefig(os.path.join(self.model.model_dir, 'precision_recall_curve.png'))
    plt.close()

def analyze_important_features(self, smiles_list=None, n_features=10):
    """
    Analyze important molecular features for predictions.
    
    Note: This is a placeholder method as feature importance analysis
    requires model-specific implementations.
    
    Parameters
    ----------
    smiles_list : list, optional
        List of SMILES strings to analyze. If None, uses test set.
    n_features : int
        Number of top features to return
    
    Returns
    -------
    dict
        Dictionary of important features and their scores
    """
    print("Feature importance analysis is not implemented for this model type.")
    print("Consider using model-specific attribution methods or SHAP values.")
    
    return {}

def save_model(self, filename=None):
    """
    Save the trained model to disk.
    
    Parameters
    ----------
    filename : str, optional
        Path to save the model. If None, uses default path.
    
    Returns
    -------
    str
        Path to the saved model
    """
    if self.model is None:
        raise ValueError("No model to save. Train the model first.")
    
    if filename is None:
        filename = os.path.join(self.model.model_dir, 'final_model')
    
    self.model.save_checkpoint(model_dir=filename)
    print(f"Model saved to {filename}")
    return filename

def load_model(self, filename):
    """
    Load a previously trained model from disk.
    
    Parameters
    ----------
    filename : str
        Path to the saved model
    
    Returns
    -------
    self
        The pipeline object with loaded model
    """
    if self.tasks is None:
        raise ValueError("Tasks not defined. Load data first.")
    
    # Rebuild the model structure
    self.build_model()
    
    # Restore weights
    self.model.restore(model_dir=filename)
    print(f"Model loaded from {filename}")
    return self

def export_predictions(self, smiles_list, output_file):
    """
    Export model predictions to a CSV file.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings to make predictions for
    output_file : str
        Path to save predictions
    
    Returns
    -------
    str
        Path to the saved predictions
    """
    if self.model is None:
        raise ValueError("No model available. Train or load a model first.")
    
    # Make predictions
    preds = self.predict(smiles_list)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'smiles': smiles_list
    })
    
    # Add predictions for each task
    for i, task in enumerate(self.tasks):
        df[f'{task}_score'] = preds[:, i]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    return output_file

def screen_virtual_library(self, library_file, output_file=None, conf_threshold=0.5):
    """
    Screen a virtual library of compounds and rank them by prediction scores.
    
    Parameters
    ----------
    library_file : str
        Path to CSV or SDF file containing compounds to screen
    output_file : str, optional
        Path to save screening results
    conf_threshold : float
        Confidence threshold for binary classification
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with screening results
    """
    if self.model is None:
        raise ValueError("No model available. Train or load a model first.")
    
    # Load compounds
    if library_file.endswith('.csv'):
        df = pd.read_csv(library_file)
        if 'smiles' in df.columns:
            smiles_list = df['smiles'].tolist()
        else:
            raise ValueError("CSV file must contain a 'smiles' column")
    elif library_file.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(library_file)
        smiles_list = [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
    else:
        raise ValueError("Unsupported file format. Use CSV or SDF.")
    
    print(f"Screening {len(smiles_list)} compounds...")
    
    # Make predictions
    preds = self.predict(smiles_list)
    
    # Create a DataFrame
    results = pd.DataFrame({
        'smiles': smiles_list
    })
    
    # Add predictions for each task
    for i, task in enumerate(self.tasks):
        results[f'{task}_score'] = preds[:, i]
        
        # For classification, add binary prediction
        if self.model.mode == "classification":
            results[f'{task}_active'] = (preds[:, i] >= conf_threshold).astype(int)
    
    # Sort by prediction score (descending)
    results = results.sort_values(by=f'{self.tasks[0]}_score', ascending=False)
    
    # Save results if output file is provided
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"Screening results saved to {output_file}")
    
    return results
    def run_full_pipeline(self, training_file, external_file=None, output_dir="./output", 
                     n_active_learning_cycles=3, oracle_function=None):
      """
      Run the full drug discovery pipeline from data loading to active learning.
      
      Parameters
      ----------
      training_file : str
          Path to training data file (CSV or SDF)
      external_file : str, optional
          Path to external compounds for active learning
      output_dir : str
          Directory to save results
      n_active_learning_cycles : int
          Number of active learning cycles to run
      oracle_function : callable, optional
          Function to simulate experimental validation
      
      Returns
      -------
      self
          The pipeline instance
      """
      # Create output directory
      os.makedirs(output_dir, exist_ok=True)
      model_dir = os.path.join(output_dir, "model")
      
      # Load and preprocess data
      self.load_data(compound_file=training_file)
      
      # Build and train the model
      self.build_model(model_dir=model_dir)
      self.train_model(epochs=100, patience=10)
      
      # Evaluate the model
      self.evaluate_model()
      self.plot_roc_curve()
      self.plot_precision_recall_curve()
      
      # Save the model
      self.save_model()
      
      # Setup active learning if enabled
      if self.active_learning and external_file:
          self.setup_active_learning(external_file, batch_size=10)
          self.run_active_learning_cycle(n_cycles=n_active_learning_cycles, 
                                        oracle_function=oracle_function)
      
      print("Pipeline completed successfully!")
      return self
