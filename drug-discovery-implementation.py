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
from keras import backend as K
from keras.models import load_model

# New: Import necessary libraries for Google Drive access
from google.colab import drive

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

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
        # Initialize tasks at creation time
        self.tasks = ["activity"]

    def load_data(self, smiles_list=None, labels=None, compound_file=None, test_size=0.2, valid_size=0.1):
        """Load and preprocess the dataset."""
        if smiles_list is None and compound_file is None:
            raise ValueError("Provide either smiles_list or compound_file")
        
        # Set tasks at the beginning to ensure it's defined
        self.tasks = ["activity"]
        print(f"Setting tasks to: {self.tasks}")
        
        # Load compounds from file
        if compound_file:
            print(f"Attempting to load from file: {compound_file}")
            
            # Check if file exists
            if not os.path.exists(compound_file):
                raise ValueError(f"File not found: {compound_file}")
                
            if compound_file.endswith('.csv'):
                try:
                    df = pd.read_csv(compound_file)
                    print(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
                    print(f"First few rows:\n{df.head()}")
                    
                    if 'smiles' in df.columns and 'label' in df.columns:
                        smiles_list = df['smiles'].tolist()
                        labels = df['label'].tolist()
                        # Convert labels to a numpy array of shape (n_samples, 1)
                        labels = np.array(labels)
                        labels = labels.reshape(-1, 1)
                        print(f"Found {len(smiles_list)} compounds with labels")
                    else:
                        raise ValueError(f"CSV file must contain 'smiles' and 'label' columns. Found: {df.columns.tolist()}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"Error reading CSV file: {e}")

            elif compound_file.endswith('.sdf'):
                try:
                    suppl = Chem.SDMolSupplier(compound_file)
                    mols = [mol for mol in suppl if mol is not None]
                    smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
                    # Extract properties or labels from SDF if available
                    labels = [int(mol.GetProp('activity')) if mol.HasProp('activity') else 0 for mol in mols]
                    labels = np.array(labels)
                    labels = labels.reshape(-1, 1)
                except Exception as e:
                    raise ValueError(f"Error reading SDF file: {e}")
            else:
                raise ValueError("Unsupported file format. Use CSV or SDF.")
        else:
            if labels is None:
                raise ValueError("Provide label list and smiles list.")
        print(f"Loading custom dataset...")

        # Confirm tasks are set before proceeding
        print(f"Tasks before featurization: {self.tasks}")

        # Choose featurizer
        featurizer = MolGraphConvFeaturizer()

        # Featurize the molecules
        try:
            features = featurizer.featurize(smiles_list)
            print(f"Featurized {len(features)} molecules successfully")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error featurizing molecules: {e}")

        # Convert to numpy arrays if needed
        X = np.array([f.adj for f in features]) # Properly extract adjacency matrices
        y = np.array(labels)
        ids = np.array(smiles_list)  # Use SMILES as IDs

        # Split data
        if self.split_type == "scaffold" and len(smiles_list) > 10:
            # For scaffold splitting, first convert to molecules
            try:
                mols = [Chem.MolFromSmiles(s) for s in smiles_list]
                splitter = ScaffoldSplitter()
                train_idx, valid_idx, test_idx = splitter.split(mols, frac_train=0.7, frac_valid=0.15, frac_test=0.15)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise ValueError(f"Error performing scaffold split: {e}")
            # Create datasets based on indices
            train_X = X[train_idx]
            valid_X = X[valid_idx]
            test_X = X[test_idx]
            train_y = y[train_idx]
            valid_y = y[valid_idx]
            test_y = y[test_idx]
            train_ids = ids[train_idx]
            valid_ids = ids[valid_idx]
            test_ids = ids[test_idx]

            self.train_dataset = NumpyDataset(train_X, train_y, ids=train_ids)
            self.valid_dataset = NumpyDataset(valid_X, valid_y, ids=valid_ids)
            self.test_dataset = NumpyDataset(test_X, test_y, ids=test_ids)
        else:
            # Random split
            indices = list(range(len(smiles_list)))
            train_idx, temp_idx = train_test_split(indices, test_size=test_size, random_state=42)
            valid_idx, test_idx = train_test_split(temp_idx, test_size=valid_size / (test_size + valid_size), random_state=42)

            # Create DeepChem datasets
            train_X = X[train_idx]
            valid_X = X[valid_idx]
            test_X = X[test_idx]
            train_y = y[train_idx]
            valid_y = y[valid_idx]
            test_y = y[test_idx]
            train_ids = ids[train_idx]
            valid_ids = ids[valid_idx]
            test_ids = ids[test_idx]

            self.train_dataset = NumpyDataset(train_X, train_y, ids=train_ids)
            self.valid_dataset = NumpyDataset(valid_X, valid_y, ids=valid_ids)
            self.test_dataset = NumpyDataset(test_X, test_y, ids=test_ids)


        print(f"Dataset loaded with {len(self.train_dataset)} training samples,",
              f"{len(self.valid_dataset)} validation samples, and {len(self.test_dataset)} test samples.")
        print(f"Tasks after data loading: {self.tasks}")

        # Analyze dataset
        self._analyze_dataset()

        return self

    def _analyze_dataset(self):
        """Analyze the dataset and print statistics."""
        # Count positives and negatives in training set
        if self.train_dataset is None or len(self.train_dataset) == 0:
            print("Warning: Training dataset is empty.  Cannot analyze.")
            return

        y_train = self.train_dataset.y
        n_tasks = len(self.tasks)

        print("\nDataset statistics:")

        # Check if y_train has the expected shape (n_samples, n_tasks)
        if y_train.ndim == 1:
             print("Reshaping y_train to (n_samples, 1)")
             y_train = y_train.reshape(-1, 1)  # Reshape to (n_samples, 1)

        for i, task in enumerate(self.tasks):
            # Add a check to prevent IndexError
            if i >= y_train.shape[1]:
                print(f"Warning: Task index {i} is out of bounds for y_train. Skipping task.")
                continue  # Skip to the next task
            positives = np.sum(y_train[:, i] == 1)
            negatives = np.sum(y_train[:, i] == 0)
            print(f"Task '{task}': {positives} positives, {negatives} negatives")

    def build_model(self, n_tasks=None, model_dir="./model",
                    batch_size=128, learning_rate=0.001, n_layers=2):
        """
        Build and compile a Graph Convolutional Network model.
        """
        # Double-check tasks is defined
        if self.tasks is None:
            print("WARNING: Tasks is None in build_model(), setting default")
            self.tasks = ["activity"]
            
        print(f"Building model with tasks: {self.tasks}")

        if n_tasks is None:
            n_tasks = len(self.tasks)
            
        print(f"Using n_tasks = {n_tasks}")

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
            raise ValueError("Model has not been built yet. Call load_data() first.")

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
            try:
                df = pd.read_csv(external_compounds_file)
                if 'smiles' in df.columns:
                    smiles_list = df['smiles'].tolist()
                else:
                    raise ValueError("CSV file must contain a 'smiles' column")
            except Exception as e:
                raise ValueError(f"Error reading external CSV file: {e}")
        elif external_compounds_file.endswith('.sdf'):
            try:
                suppl = Chem.SDMolSupplier(external_compounds_file)
                smiles_list = [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
            except Exception as e:
                raise ValueError(f"Error reading external SDF file: {e}")

        else:
            raise ValueError("Unsupported file format. Use CSV or SDF.")

        print(f"Loaded {len(smiles_list)} external compounds for screening.")

        # Featurize the compounds
        featurizer = MolGraphConvFeaturizer()

        try:
            X = featurizer.featurize(smiles_list)
        except Exception as e:
            raise ValueError(f"Error featurizing external molecules: {e}")
        y = np.zeros((len(X), len(self.tasks)))  # Initialize with placeholder labels
        ids = np.array(smiles_list)

        # Create unlabeled pool
        self.unlabeled_pool = NumpyDataset(X, y, ids=ids)
        print(f"Created unlabeled pool with {len(self.unlabeled_pool)} compounds.")

        # Create initial active learning set - ensuring we don't exceed the dataset size
        initial_size = min(initial_size, len(self.train_dataset))

        # Extract the initial training subset
        initial_X = self.train_dataset.X[:initial_size]
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
                selected_X = self.unlabeled_pool.X[selected_indices]
                selected_y = np.array(new_labels).reshape(-1, len(self.tasks))
                selected_ids = np.array(selected_smiles)

                new_data = NumpyDataset(selected_X, selected_y, ids=selected_ids)

                # Merge new data with active learning set
                self.active_learning_set = dc.data.NumpyDataset.merge([self.active_learning_set, new_data])

                # Remove selected compounds from unlabeled pool
                mask = np.ones(len(self.unlabeled_pool), dtype=bool)
                mask[selected_indices] = False

                X_remaining = self.unlabeled_pool.X[mask]
                y_remaining = self.unlabeled_pool.y[mask]
                ids_remaining = self.unlabeled_pool.ids[mask]

                self.unlabeled_pool = NumpyDataset(X_remaining, y_remaining, ids=ids_remaining)

                print(f"Unlabeled_pool size reduced to {len(self.unlabeled_pool)} samples.")

            else:
                print("Oracle function not provided. Skipping labeling and updating active learning set.")

            # Store active learning history (optional)
            self.active_learning_history.append({
                'cycle': cycle + 1,
                'selected_smiles': selected_smiles,
                'test_score': np.mean(list(test_score.values()))
            })

        return self

    def _select_compounds(self, batch_size):
        """Select compounds from the unlabeled pool using uncertainty sampling."""
        if not self.active_learning:
            raise ValueError("Active learning is disabled.")
        if self.unlabeled_pool is None:
            raise ValueError("Unlabeled pool has not been initialized.")

        # Predict probabilities for all compounds in the unlabeled pool
        predictions = self.model.predict(self.unlabeled_pool)

        # Uncertainty sampling: select compounds with probabilities closest to 0.5
        if self.model.mode == "classification":
            uncertainties = np.abs(predictions - 0.5)
        else:  # Regression
            uncertainties = np.abs(predictions - np.mean(predictions))

        # Rank compounds by uncertainty
        ranked_indices = np.argsort(uncertainties.flatten())

        # Select the top batch_size most uncertain compounds
        selected_indices = ranked_indices[:batch_size]
        print(f"Selected {len(selected_indices)} compounds based on uncertainty.")

        return selected_indices

    def visualize_compounds(self, smiles_list, filename="compounds.png", mols_per_row=5):
        """Visualize a list of compounds and save the image."""
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(200, 200))
        img.save(filename)
        print(f"Compound visualization saved to {filename}")
        return self

    def test_model(self):
        """
        Test the model on the test dataset and print the performance metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        # Define metric
        if self.model.mode == "classification":
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        else:
            metric = dc.metrics.Metric(dc.metrics.r2_score)

        test_score = self.model.evaluate(self.test_dataset, [metric])
        print(f"Test score: {np.mean(list(test_score.values())):.4f}")

        # Generate plots if classification
        if self.model.mode == "classification":
            # Get predictions on the test set
            preds = self.model.predict(self.test_dataset)
            y_true = self.test_dataset.y.flatten()  # Ground truth labels
            y_scores = preds.flatten()

            # ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.model.model_dir, 'roc_curve.png'))

            # Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(self.model.model_dir, 'pr_curve.png'))

        return self

# Mount Google Drive
drive.mount('/content/drive')

# Example usage: Replace with your actual file path
file_path = '/content/drive/MyDrive/bace(1).csv'  # Or your .sdf file

# Check if file exists before proceeding
if not os.path.exists(file_path):
    print(f"ERROR: File not found: {file_path}")
    print("Available files in directory:")
    if os.path.exists('/content/drive/MyDrive'):
        print(os.listdir('/content/drive/MyDrive'))
    exit()
else:
    print(f"File found: {file_path}")
    
    # Check file contents
    if file_path.endswith('.csv'):
        try:
            test_df = pd.read_csv(file_path)
            print(f"CSV preview:\n{test_df.head()}")
            print(f"Columns: {test_df.columns.tolist()}")
        except Exception as e:
            print(f"Error previewing CSV: {e}")
            exit()

# Instantiate the pipeline with explicit initialization of tasks
pipeline = DrugDiscoveryPipeline(dataset_name="MyDataset", featurizer_type="graph_conv", split_type="scaffold", active_learning=False)
print(f"Pipeline created with tasks: {pipeline.tasks}")

# Load the data from Google Drive with improved error handling
try:
    pipeline.load_data(compound_file=file_path)
    print(f"After loading, tasks = {pipeline.tasks}")
    
    # Verify tasks are set
    if pipeline.tasks is None:
        print("WARNING: Tasks is still None after loading data, setting it manually")
        pipeline.tasks = ["activity"]
    
    # Build the model
    pipeline.build_model(n_tasks=1, model_dir="./my_model", batch_size=128, learning_rate=0.001, n_layers=2)
    
    # Train the model
    pipeline.train_model(epochs=10, patience=3)
    
    # Test the model
    pipeline.test_model()
    
except Exception as e:
    print(f"ERROR during pipeline execution: {str(e)}")
    import traceback
    traceback.print_exc()