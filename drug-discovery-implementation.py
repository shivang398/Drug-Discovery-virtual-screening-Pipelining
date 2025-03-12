"""
Drug Discovery Virtual Screening Pipeline using DeepChem
========================================================
This project implements an end-to-end pipeline for virtual screening
of potential drug candidates with active learning for compound prioritization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import deepchem as dc
from deepchem.models import GraphConvModel
from deepchem.feat import ConvMolFeaturizer, MolGraphConvFeaturizer
from deepchem.splits import RandomSplitter, ScaffoldSplitter
from deepchem.data import NumpyDataset
import tensorflow as tf

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
            Type of featurizer to use ('graph_conv', 'conv_mol', etc.)
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
        
    def load_data(self):
        """Load and preprocess the dataset."""
        print(f"Loading {self.dataset_name} dataset...")
        
        # Choose featurizer based on user selection
        if self.featurizer_type == "graph_conv":
            featurizer = MolGraphConvFeaturizer()
        elif self.featurizer_type == "conv_mol":
            featurizer = ConvMolFeaturizer()
        else:
            raise ValueError(f"Unsupported featurizer: {self.featurizer_type}")
        
        # Load the dataset
        loader = dc.molnet.load_dataset(
            self.dataset_name, featurizer=featurizer, splitter=self.split_type
        )
        
        # Unpack the loader
        tasks, datasets, transformers = loader
        self.tasks = tasks
        self.train_dataset, self.valid_dataset, self.test_dataset = datasets
        self.transformers = transformers
        
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
        
        Parameters
        ----------
        n_tasks : int, optional
            Number of tasks (targets) to predict
        model_dir : str
            Directory to save the model to
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for the optimizer
        n_layers : int
            Number of graph convolutional layers
        """
        if n_tasks is None:
            n_tasks = len(self.tasks)
        
        # Determine the model type (classification or regression)
        model_type = "classification"
        if "regression" in self.dataset_name:
            model_type = "regression"
            
        # Build the model
        print(f"Building GraphConvModel with {n_tasks} tasks, mode={model_type}")
        self.model = GraphConvModel(
            n_tasks=n_tasks,
            mode=model_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_dir=model_dir,
            graph_conv_layers=[64] * n_layers,
            dense_layer_size=128,
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
        if "classification" in self.model.model_dir or any("class" in t for t in self.tasks):
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
            self.model.fit(self.train_dataset, nb_epoch=1)
            
            # Evaluate
            train_score = self.model.evaluate(self.train_dataset, [metric])
            valid_score = self.model.evaluate(self.valid_dataset, [metric])
            
            # Print progress
            mean_train = np.mean(list(train_score.values()))
            mean_valid = np.mean(list(valid_score.values()))
            train_scores.append(mean_train)
            valid_scores.append(mean_valid)
            
            print(f"Epoch {epoch+1}/{epochs}: train = {mean_train:.4f}, valid = {mean_valid:.4f}")
            
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
                    print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
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
            
        print("Setting up active learning pipeline...")
        
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
        if self.featurizer_type == "graph_conv":
            featurizer = MolGraphConvFeaturizer()
        else:
            featurizer = ConvMolFeaturizer()
            
        X = featurizer.featurize(smiles_list)
        y = np.zeros((len(X), len(self.tasks)))  # Initialize with placeholder labels
        ids = np.array(smiles_list)
        
        # Create unlabeled pool
        self.unlabeled_pool = NumpyDataset(X, y, ids=ids)
        print(f"Created unlabeled pool with {len(self.unlabeled_pool)} compounds.")
        
        # Create initial active learning set
        self.active_learning_set = NumpyDataset(
            self.train_dataset.X[:initial_size],
            self.train_dataset.y[:initial_size],
            ids=self.train_dataset.ids[:initial_size]
        )
        
        self.al_batch_size = batch_size
        print(f"Active learning initialized with {initial_size} compounds and batch size {batch_size}.")
        
        return self
    
    def run_active_learning(self, n_iterations=10, uncertainty_method="entropy"):
        """
        Run the active learning pipeline.
        
        Parameters
        ----------
        n_iterations : int
            Number of active learning iterations to run
        uncertainty_method : str
            Method to calculate uncertainty ('entropy', 'margin', 'least_confident')
        """
        if not self.active_learning or self.unlabeled_pool is None:
            raise ValueError("Active learning not set up. Call setup_active_learning() first.")
        
        print(f"Running active learning for {n_iterations} iterations...")
        
        # Define metric
        if "classification" in self.model.model_dir or any("class" in t for t in self.tasks):
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        else:
            metric = dc.metrics.Metric(dc.metrics.r2_score)
        
        # Run active learning iterations
        for iteration in range(n_iterations):
            print(f"\nActive Learning Iteration {iteration+1}/{n_iterations}")
            
            # Retrain model on current active learning set
            self.model.fit(self.active_learning_set, nb_epoch=10)
            
            # Evaluate model
            valid_score = self.model.evaluate(self.valid_dataset, [metric])
            mean_valid = np.mean(list(valid_score.values()))
            print(f"Validation score: {mean_valid:.4f}")
            
            # Make predictions on unlabeled pool
            y_pred = self.model.predict(self.unlabeled_pool)
            
            # Calculate uncertainty based on selected method
            if uncertainty_method == "entropy":
                # For classification, use entropy of prediction probabilities
                uncertainty = -np.sum(y_pred * np.log(y_pred + 1e-10), axis=2).mean(axis=1)
            elif uncertainty_method == "margin":
                # Difference between top two class probabilities
                y_pred_sorted = np.sort(y_pred, axis=2)[:, :, ::-1]
                uncertainty = (y_pred_sorted[:, :, 0] - y_pred_sorted[:, :, 1]).mean(axis=1)
                uncertainty = -uncertainty  # Higher is more uncertain
            elif uncertainty_method == "least_confident":
                # 1 - max probability
                uncertainty = 1 - np.max(y_pred, axis=2).mean(axis=1)
            else:
                raise ValueError(f"Unsupported uncertainty method: {uncertainty_method}")
            
            # Select most uncertain samples
            indices = np.argsort(uncertainty)[-self.al_batch_size:]
            
            # In a real scenario, you would get labels for these compounds
            # Here we'll simulate it by using the model's predictions
            selected_X = self.unlabeled_pool.X[indices]
            selected_ids = self.unlabeled_pool.ids[indices]
            
            # Simulate getting "true" labels (in reality would come from experiments)
            # For demonstration, we'll use model predictions as proxy
            # In real application, this would be replaced with wet lab testing results
            selected_y = y_pred[indices].mean(axis=2)
            
            # Add selected compounds to active learning set
            new_X = np.vstack([self.active_learning_set.X, selected_X])
            new_y = np.vstack([self.active_learning_set.y, selected_y])
            new_ids = np.concatenate([self.active_learning_set.ids, selected_ids])
            
            # Update active learning set
            self.active_learning_set = NumpyDataset(new_X, new_y, ids=new_ids)
            
            # Remove selected compounds from unlabeled pool
            mask = np.ones(len(self.unlabeled_pool), dtype=bool)
            mask[indices] = False
            self.unlabeled_pool = NumpyDataset(
                self.unlabeled_pool.X[mask],
                self.unlabeled_pool.y[mask],
                ids=self.unlabeled_pool.ids[mask]
            )
            
            # Track progress
            self.active_learning_history.append({
                'iteration': iteration + 1,
                'validation_score': mean_valid,
                'selected_compounds': selected_ids.tolist()
            })
            
            print(f"Selected {len(indices)} compounds. Active learning set now has {len(self.active_learning_set)} compounds.")
            print(f"Unlabeled pool has {len(self.unlabeled_pool)} compounds remaining.")
        
        # Final evaluation
        self.model.fit(self.active_learning_set, nb_epoch=20)
        final_score = self.model.evaluate(self.test_dataset, [metric])
        print(f"\nFinal test score after active learning: {np.mean(list(final_score.values())):.4f}")
        
        # Plot active learning progress
        plt.figure(figsize=(10, 6))
        plt.plot([h['iteration'] for h in self.active_learning_history],
                [h['validation_score'] for h in self.active_learning_history],
                'o-')
        plt.xlabel('Active Learning Iteration')
        plt.ylabel('Validation Score')
        plt.title('Active Learning Progress')
        plt.grid(True)
        plt.savefig(os.path.join(self.model.model_dir, 'active_learning_progress.png'))
        
        return self
    
    def screen_compounds(self, smiles_list=None, compound_file=None, top_k=100):
        """
        Screen compounds using the trained model.
        
        Parameters
        ----------
        smiles_list : list of str, optional
            List of SMILES strings to screen
        compound_file : str, optional
            Path to a file containing compounds to screen (CSV or SDF)
        top_k : int
            Number of top compounds to return
            
        Returns
        -------
        pd.DataFrame
            DataFrame with screening results
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
            
        # Load compounds
        if smiles_list is None and compound_file is None:
            raise ValueError("Must provide either smiles_list or compound_file")
            
        if compound_file is not None:
            if compound_file.endswith('.csv'):
                df = pd.read_csv(compound_file)
                if 'smiles' in df.columns:
                    smiles_list = df['smiles'].tolist()
                else:
                    raise ValueError("CSV file must contain a 'smiles' column")
            elif compound_file.endswith('.sdf'):
                suppl = Chem.SDMolSupplier(compound_file)
                smiles_list = [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
            else:
                raise ValueError("Unsupported file format. Use CSV or SDF.")
        
        print(f"Screening {len(smiles_list)} compounds...")
        
        # Featurize compounds
        if self.featurizer_type == "graph_conv":
            featurizer = MolGraphConvFeaturizer()
        else:
            featurizer = ConvMolFeaturizer()
            
        X = featurizer.featurize(smiles_list)
        
        # Create dataset
        dataset = NumpyDataset(X, np.zeros((len(X), len(self.tasks))), ids=np.array(smiles_list))
        
        # Make predictions
        y_pred = self.model.predict(dataset)
        
        # Convert predictions to DataFrame
        results = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Calculate molecular properties
            mw = Chem.Descriptors.MolWt(mol)
            logp = Chem.Descriptors.MolLogP(mol)
            tpsa = Chem.Descriptors.TPSA(mol)
            h_donors = Chem.Descriptors.NumHDonors(mol)
            h_acceptors = Chem.Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Chem.Descriptors.NumRotatableBonds(mol)
            
            # Get predictions for each task
            preds = {}
            for j, task in enumerate(self.tasks):
                pred_value = y_pred[i][0][j]  # Get prediction value
                preds[f"{task}_prediction"] = pred_value
            
            # Combine all information
            result = {
                'smiles': smiles,
                'molecular_weight': mw,
                'logP': logp,
                'TPSA': tpsa,
                'h_donors': h_donors,
                'h_acceptors': h_acceptors,
                'rotatable_bonds': rotatable_bonds,
                **preds
            }
            results.append(result)
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Sort by the first task prediction (or average if multiple tasks)
        if len(self.tasks) > 1:
            pred_cols = [f"{task}_prediction" for task in self.tasks]
            df_results['avg_prediction'] = df_results[pred_cols].mean(axis=1)
            df_results = df_results.sort_values('avg_prediction', ascending=False)
        else:
            df_results = df_results.sort_values(f"{self.tasks[0]}_prediction", ascending=False)
        
        # Apply Lipinski's Rule of Five filter
        lipinski_violations = (
            (df_results['molecular_weight'] > 500).astype(int) +
            (df_results['logP'] > 5).astype(int) +
            (df_results['h_donors'] > 5).astype(int) +
            (df_results['h_acceptors'] > 10).astype(int)
        )
        df_results['lipinski_violations'] = lipinski_violations
        
        # Get top compounds
        top_compounds = df_results.head(top_k)
        
        # Print top 10 compounds
        print("\nTop 10 compounds:")
        for i, (_, row) in enumerate(top_compounds.head(10).iterrows()):
            task_preds = ', '.join([f"{task}: {row[f'{task}_prediction']:.3f}" for task in self.tasks])
            print(f"{i+1}. {row['smiles']} - {task_preds}")
        
        # Save results
        result_path = os.path.join(self.model.model_dir, 'screening_results.csv')
        df_results.to_csv(result_path, index=False)
        print(f"Full results saved to {result_path}")
        
        # Visualize top molecules
        self._visualize_top_compounds(top_compounds.head(10))
        
        return top_compounds
    
    def _visualize_top_compounds(self, top_compounds, n=10):
        """Visualize the top compounds."""
        mols = [Chem.MolFromSmiles(smiles) for smiles in top_compounds['smiles']]
        mols = [mol for mol in mols if mol is not None]
        
        # Generate labels
        if len(self.tasks) > 1:
            labels = [f"#{i+1} ({s['avg_prediction']:.2f})" for i, s in enumerate(top_compounds.to_dict('records'))]
        else:
            task = self.tasks[0]
            labels = [f"#{i+1} ({s[f'{task}_prediction']:.2f})" for i, s in enumerate(top_compounds.to_dict('records'))]
        
        # Draw molecules
        img = Draw.MolsToGridImage(mols[:n], molsPerRow=5, subImgSize=(200, 200), legends=labels)
        
        # Save image
        img_path = os.path.join(self.model.model_dir, 'top_compounds.png')
        img.save(img_path)
        print(f"Visualization of top compounds saved to {img_path}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DrugDiscoveryPipeline(
        dataset_name="bace_classification",
        featurizer_type="graph_conv",
        split_type="scaffold",
        active_learning=True
    )
    
    # Load data
    pipeline.load_data()
    
    # Build and train model
    pipeline.build_model(model_dir="./bace_model")
    pipeline.train_model(epochs=50)
    
    # Setup and run active learning
    # In a real scenario, you would provide a file with external compounds
    # pipeline.setup_active_learning("external_compounds.csv")
    # pipeline.run_active_learning(n_iterations=5)
    
    # Screen compounds
    # In a real scenario, you would provide compounds to screen
    # top_compounds = pipeline.screen_compounds(compound_file="compounds_to_screen.csv")
