import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Check if running in Google Colab
def check_if_colab():
    """Check if the code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_google_drive():
    """Mount Google Drive to access data if running in Colab."""
    if check_if_colab():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully.")
            return True
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
            return False
    else:
        print("Not running in Google Colab. Skipping Google Drive mount.")
        return False

def get_file_path_from_drive(file_name=None, folder_path=None):
    """
    Construct a file path from Google Drive or prompt user to provide one.
    """
    if check_if_colab():
        # Base directory in Google Drive
        base_path = '/content/drive/MyDrive/bace(1).csv'

        # Construct path with optional folder
        if folder_path:
            base_path = os.path.join(base_path, folder_path)

        # Add file name if provided, otherwise prompt user
        if file_name:
            return os.path.join(base_path, file_name)
        else:
            from IPython.display import display, HTML
            display(HTML("<p style='color: orange'>Please enter the path to your data file in Google Drive (relative to MyDrive):</p>"))
            user_path = input("Enter file path (e.g., 'data/compounds.csv'): ")
            return os.path.join('/content/drive/MyDrive/', user_path)
    else:
        if file_name:
            return file_name
        else:
            return input("Enter file path: ")
# Define the MolecularDescriptorCalculator class here, before DrugDiscoveryPipeline
class MolecularDescriptorCalculator:
    """
    Calculates fingerprints for a list of molecules.
    """

    def __init__(self, fp_radius=2, fp_bits=1024):
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

    def calculate_for_mols(self, mols):
        """
        Calculate fingerprints for a list of RDKit molecules.

        Parameters
        ----------
        mols : list
            List of RDKit Mol objects.

        Returns
        -------
        np.ndarray
            2D array of calculated fingerprints.
        """
        # Calculate fingerprints
        fingerprints = [
            AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
            for mol in mols
        ]
        fingerprints_array = np.array([list(fp) for fp in fingerprints], dtype=int)

        return fingerprints_array


class MolecularDataAugmentation:
    """
    Class for performing data augmentation on molecular datasets.
    Provides methods to generate synthetic molecular variants.
    """

    @staticmethod
    def apply_transformations(mol, num_variants=5):
        """
        Generate multiple molecular variants through different transformations.

        Parameters
        ----------
        mol : RDKit Mol
            Input molecule to transform
        num_variants : int
            Number of variants to generate

        Returns
        -------
        List[RDKit Mol]
            List of transformed molecules
        """
        if mol is None:
            return []

        variants = []

        # Canonical variant (original molecule)
        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        variants.append(canonical_mol)

        # Different sanitization and standardization techniques
        enumerator = rdMolStandardize.TautomerEnumerator()
        canonical_taut = enumerator.Canonicalize(mol)
        variants.append(canonical_taut)

        # Random 3D conformer generation with different parameters
        for _ in range(min(num_variants - 2, 3)):
            try:
                # Generate a 3D conformer
                conformer_mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(conformer_mol, randomSeed=np.random.randint(1000))
                AllChem.MMFFOptimizeMolecule(conformer_mol)

                # Convert back to 2D SMILES to create a new molecule
                smiles = Chem.MolToSmiles(conformer_mol)
                conf_mol = Chem.MolFromSmiles(smiles)
                variants.append(conf_mol)
            except:
                pass

        # Minimal structural modifications
        try:
            # Slight stereochemistry modification
            stereo_mol = Chem.MolFromSmiles(
                Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
            )
            variants.append(stereo_mol)
        except:
            pass

        return [v for v in variants if v is not None]

    @staticmethod
    def generate_synthetic_molecules(mols, labels, augmentation_factor=2):
        """
        Generate synthetic molecules through augmentation.

        Parameters
        ----------
        mols : List[RDKit Mol]
            Original molecules
        labels : np.ndarray
            Corresponding labels
        augmentation_factor : int
            How many variants to generate per original molecule

        Returns
        -------
        Tuple[List[RDKit Mol], np.ndarray]
            Augmented molecules and their corresponding labels
        """
        augmented_mols = []
        augmented_labels = []

        for mol, label in zip(mols, labels):
            # Original molecule
            augmented_mols.append(mol)
            augmented_labels.append(label)

            # Generate variants
            variants = MolecularDataAugmentation.apply_transformations(
                mol, num_variants=augmentation_factor
            )

            for variant in variants[1:]:  # Skip the first variant (original)
                augmented_mols.append(variant)
                augmented_labels.append(label)

        return augmented_mols, np.array(augmented_labels)






class DrugDiscoveryPipeline:
    """Extension of DrugDiscoveryPipeline with 70/30 data split and anti-overfitting measures"""
    def __init__(self, dataset_name: str = "bace_classification",
                 split_type: str = "scaffold",
                 active_learning: bool = True,
                 fp_radius: int = 2,
                 fp_bits: int = 2048):
        """
        Initialize the drug discovery pipeline.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to use
        split_type : str
            Type of data split ('random', 'scaffold')
        active_learning : bool
            Whether to use active learning for compound selection
        fp_radius : int
            Radius for Morgan fingerprints
        fp_bits : int
            Number of bits for Morgan fingerprints
        """
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.active_learning = active_learning
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

        self.model = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.unlabeled_pool = None
        self.active_learning_history = []
        self.tasks = ["activity"]

        # Initialize feature generators
        self.descriptor_calculator = MolecularDescriptorCalculator(
            fp_radius=fp_radius,
            fp_bits=fp_bits
        )

        self.feature_scaler = None

    def load_data(self, smiles_list=None, labels=None, compound_file=None,
                  test_size=0.3, smiles_column='mol', label_column='Class'):
        """
        Load and preprocess the dataset with 70/30 train/test split.

        Parameters
        ----------
        smiles_list : List[str], optional
            List of SMILES strings
        labels : np.ndarray, optional
            Array of labels
        compound_file : str, optional
            File path to compound data (CSV or SDF)
        test_size : float
            Fraction of data to use for testing (default: 0.3 for 70/30 split)
        smiles_column : str
            Column name containing SMILES in CSV file
        label_column : str
            Column name containing labels in CSV file

        Returns
        -------
        self
            Returns self for method chaining
        """
        if smiles_list is None and compound_file is None:
            raise ValueError("Provide either smiles_list or compound_file")

        # Load compounds from file
        if compound_file:
            print(f"Loading data from file: {compound_file}")

            # Check if file exists
            if not os.path.exists(compound_file):
                raise ValueError(f"File not found: {compound_file}")

            if compound_file.endswith('.csv'):
                try:
                    df = pd.read_csv(compound_file)
                    print(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
                    print(f"First few rows:\n{df.head()}")

                    # Check if specified columns exist
                    if smiles_column in df.columns and label_column in df.columns:
                        smiles_list = df[smiles_column].tolist()
                        labels = df[label_column].values

                        # Attempt to convert labels to binary (0 and 1)
                        # Replace string values with numeric
                        labels = np.where(labels == 'Inactive', 0, labels)
                        labels = np.where(labels == 'Active', 1, labels)

                        # Convert to numeric, handling errors by setting to 0
                        # Original: labels = pd.to_numeric(labels, errors='coerce').fillna(0).astype(int)
                        # Fix: Use np.nan_to_num instead of fillna for numpy arrays
                        labels = pd.to_numeric(labels, errors='coerce')

                        # Check if the array is a Series or not
                        if isinstance(labels, pd.Series):
                          labels = labels.fillna(0).astype(int)
                        else:
                          labels = np.nan_to_num(labels, nan=0).astype(int)


                        # Check if labels are binary after conversion
                        if not np.all(np.isin(labels, [0, 1])):
                            raise ValueError(
                                "Target variable ('Class') contains non-binary values even after conversion. "
                                "Please check your data."
                            )

                        # Convert labels to a numpy array of shape (n_samples, 1)
                        labels = np.array(labels).reshape(-1, 1)
                        print(f"Found {len(smiles_list)} compounds with labels")
                    else:
                        raise ValueError(
                            f"CSV file must contain '{smiles_column}' and '{label_column}' columns. "
                            f"Found: {df.columns.tolist()}"
                        )
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
                    labels = [
                        int(mol.GetProp('activity')) if mol.HasProp('activity') else 0 for mol in mols
                    ]
                    labels = np.array(labels).reshape(-1, 1)
                except Exception as e:
                    raise ValueError(f"Error reading SDF file: {e}")
            else:
                raise ValueError("Unsupported file format. Use CSV or SDF.")
        else:
            if labels is None:
                raise ValueError("Provide label list with smiles list.")

        print(f"Processing {len(smiles_list)} molecules...")

        # Convert SMILES to molecules
        mols = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(i)
            else:
                print(f"Warning: Could not parse SMILES: {smiles}")

        # Keep only valid molecules and their labels
        valid_smiles = [smiles_list[i] for i in valid_indices]
        valid_labels = labels[valid_indices] if labels is not None else None

        print(f"Successfully processed {len(mols)} valid molecules")

        # Calculate fingerprints for each molecule
        X = self.descriptor_calculator.calculate_for_mols(mols)
        y = np.array(valid_labels)

        print(f"Generated Morgan fingerprints with shape: {X.shape}")

        # Split the data into 70% training and 30% testing
        if self.split_type == "scaffold" and len(valid_smiles) > 10:
            try:
                train_idx, test_idx = self._perform_scaffold_split_70_30(mols, valid_smiles, test_size)
                # Create a small validation set from training data (20% of training data)
                train_idx, valid_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
            except Exception as e:
                print(f"Error in scaffold split: {e}. Falling back to random split.")
                train_idx, temp_idx = train_test_split(range(len(valid_smiles)), test_size=test_size, random_state=42)
                train_idx, valid_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
                test_idx = temp_idx
        else:
            # Random split
            train_idx, test_idx = train_test_split(range(len(valid_smiles)), test_size=test_size, random_state=42)
            train_idx, valid_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

        # Create the datasets
        self.train_dataset = (X[train_idx], y[train_idx], np.array(valid_smiles)[train_idx])
        self.valid_dataset = (X[valid_idx], y[valid_idx], np.array(valid_smiles)[valid_idx])
        self.test_dataset = (X[test_idx], y[test_idx], np.array(valid_smiles)[test_idx])

        # Calculate exact percentages to report
        train_pct = len(train_idx) / len(valid_smiles) * 100
        valid_pct = len(valid_idx) / len(valid_smiles) * 100
        test_pct = len(test_idx) / len(valid_smiles) * 100

        print(f"Dataset split: {len(train_idx)} training samples ({train_pct:.1f}%), "
              f"{len(valid_idx)} validation samples ({valid_pct:.1f}%), and "
              f"{len(test_idx)} test samples ({test_pct:.1f}%).")
        print(f"Primary split: {train_pct + valid_pct:.1f}% train/validation, {test_pct:.1f}% test")

        # Analyze the dataset
        self._analyze_dataset()

        return self

    def _analyze_dataset(self):
        """
        Analyze the dataset and print statistics.
        """
        if self.train_dataset is None:
            print("Warning: Training dataset is empty. Cannot analyze.")
            return

        X_train, y_train, _ = self.train_dataset

        print("\nDataset statistics:")

        # Ensure y_train has the right shape
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # Calculate class distribution
        if len(self.tasks) > 0:
            for i, task in enumerate(self.tasks):
                if i >= y_train.shape[1]:
                    continue  # Skip if task index is out of bounds
                positives = np.sum(y_train[:, i] == 1)
                negatives = np.sum(y_train[:, i] == 0)
                print(f"Task '{task}': {positives} positives, {negatives} negatives")

                # Calculate class imbalance
                if positives + negatives > 0:
                    imbalance_ratio = max(positives, negatives) / max(1, min(positives, negatives))
                    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

        # Feature statistics for fingerprints
        print("\nFingerprint statistics:")
        fp_bits = min(10, X_train.shape[1])  # Show only first 10 bits
        for i in range(fp_bits):
            bit_sum = X_train[:, i].sum()
            bit_pct = (bit_sum / len(X_train)) * 100
            print(f"Bit {i}: active in {bit_sum} molecules ({bit_pct:.1f}%)")

    def scale_features(self):
        """
        Scale features using StandardScaler to improve model performance.
        This method scales the features in train, validation, and test datasets.

        Returns
        -------
        self
            Returns self for method chaining
        """
        from sklearn.preprocessing import StandardScaler

        if self.train_dataset is None:
            raise ValueError("Data must be loaded before scaling features.")

        X_train, y_train, smiles_train = self.train_dataset
        X_valid, y_valid, smiles_valid = self.valid_dataset
        X_test, y_test, smiles_test = self.test_dataset

        # Initialize and fit the scaler on training data only
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        # Transform validation and test data using the fitted scaler
        X_valid_scaled = self.feature_scaler.transform(X_valid)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Update the datasets with scaled features
        self.train_dataset = (X_train_scaled, y_train, smiles_train)
        self.valid_dataset = (X_valid_scaled, y_valid, smiles_valid)
        self.test_dataset = (X_test_scaled, y_test, smiles_test)

        print("Features scaled using StandardScaler.")

        # Optional: Display scaling statistics
        means = self.feature_scaler.mean_
        stds = self.feature_scaler.scale_

        print(f"Feature scaling statistics (first 5 features):")
        for i in range(min(5, len(means))):
            print(f"Feature {i}: mean={means[i]:.4f}, std={stds[i]:.4f}")

        return self

    def _perform_scaffold_split_70_30(self, mols, smiles_list, test_size):
        """
        Perform a scaffold split on the dataset with 70/30 ratio.

        Parameters
        ----------
        mols : List
            List of RDKit molecules
        smiles_list : List[str]
            List of SMILES strings
        test_size : float
            Fraction of data to use for testing (e.g., 0.3)

        Returns
        -------
        Tuple[List, List]
            Indices for train and test sets
        """
        from rdkit.Chem.Scaffolds import MurckoScaffold

        # Get scaffolds for each molecule
        scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=mol) for mol in mols]

        # Group indices by scaffold
        scaffold_to_indices = {}
        for i, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_to_indices:
                scaffold_to_indices[scaffold] = []
            scaffold_to_indices[scaffold].append(i)

        # Sort scaffolds by size (number of molecules)
        scaffold_sets = [indices for scaffold, indices in scaffold_to_indices.items()]
        scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)

        # Assign molecules to train/test sets
        train_indices = []
        test_indices = []

        train_cutoff = int(len(smiles_list) * (1 - test_size))

        molecule_count = 0
        for scaffold_set in scaffold_sets:
            if molecule_count < train_cutoff:
                train_indices.extend(scaffold_set)
            else:
                test_indices.extend(scaffold_set)
            molecule_count += len(scaffold_set)

        # Ensure we have samples in each set
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError("Scaffold split resulted in empty sets.")

        return train_indices, test_indices
    def build_model(self, model_type='classification', hidden_units=(64, 32),
                dropout=0.3, learning_rate=0.001, batch_size=32, epochs=50,
                regularization_strength=0.01,  # Reduced complexity in regularization
                l1_ratio=0.1,
                feature_reduction=False,  # New parameter for optional feature reduction
                kernel_constraint=None):
        """
        Build the model with enhanced anti-overfitting strategies.

        Additional Parameters:
        ---------------------
        feature_reduction : bool, optional
            Whether to apply feature reduction using PCA
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2
        from tensorflow.keras.constraints import max_norm
        from sklearn.decomposition import PCA

        # Store parameters
        self.model_type = model_type
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.regularization_strength = regularization_strength
        self.l1_ratio = l1_ratio
        self.feature_reduction = feature_reduction

        # Get input data
        X_train, y_train, _ = self.train_dataset

        # Optional Feature Reduction using PCA
        if feature_reduction:
            print("Applying feature reduction with PCA...")
            # Determine number of components (e.g., preserve 95% variance)
            pca = PCA(n_components=0.95)  # Retain 95% of variance
            X_train = pca.fit_transform(X_train)

            # Update input dimension for the model
            input_dim = X_train.shape[1]
            print(f"Reduced features from {self.train_dataset[0].shape[1]} to {input_dim}")
        else:
            input_dim = X_train.shape[1]

        # Calculate L1 and L2 regularization strengths
        l1_strength = regularization_strength * l1_ratio
        l2_strength = regularization_strength * (1 - l1_ratio)

        # Enhanced kernel constraint
        if kernel_constraint is None:
            kernel_constraint = max_norm(3)  # Stronger constraint to prevent large weights

        # Define the model with more robust architecture
        model = Sequential()

        # Input layer with stronger regularization and batch normalization
        model.add(Dense(
            hidden_units[0],
            input_dim=input_dim,
            activation='relu',
            kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength),
            bias_regularizer=l1_l2(l1=l1_strength/2, l2=l2_strength/2),
            kernel_constraint=kernel_constraint
        ))
        model.add(BatchNormalization())  # Normalize layer activations
        model.add(Dropout(dropout))  # Prevent co-adaptation of neurons

        # Additional hidden layers with consistent regularization
        for units in hidden_units[1:]:
            model.add(Dense(
                units,
                activation='relu',
                kernel_regularizer=l1_l2(l1=l1_strength/2, l2=l2_strength/2),
                bias_regularizer=l1_l2(l1=l1_strength/4, l2=l2_strength/4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        # Output layer
        if model_type == 'classification':
            model.add(Dense(
                1,
                activation='sigmoid',
                kernel_regularizer=l1_l2(l1=l1_strength/4, l2=l2_strength/4)
            ))
            loss_function = 'binary_crossentropy'
            self.metrics = ['accuracy', 'AUC']
        else:  # regression
            model.add(Dense(
                1,
                activation='linear',
                kernel_regularizer=l1_l2(l1=l1_strength/4, l2=l2_strength/4)
            ))
            loss_function = 'mean_squared_error'
            self.metrics = ['mean_absolute_error']

        # Compile with adaptive optimizer and learning rate
        model.compile(
            optimizer=Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=True  # Use AMSGrad variant to improve convergence
            ),
            loss=loss_function,
            metrics=self.metrics
        )

        # Store the model
        self.model = model

        # Detailed logging of anti-overfitting strategies
        print("\nAnti-Overfitting Strategies:")
        print(f"- Feature Reduction: {'Yes' if feature_reduction else 'No'}")
        print(f"- L1 Regularization: {l1_strength:.6f}")
        print(f"- L2 Regularization: {l2_strength:.6f}")
        print(f"- Dropout Rate: {dropout}")
        print(f"- Batch Normalization: Applied")
        print(f"- Kernel Constraint: Max Norm = 3")

        model.summary()

        return self
    def train(self, patience=20, min_delta=0.0001):
        """
        Train the model with enhanced early stopping, learning rate reduction,
        and cyclical/adaptive learning rate scheduling.

        Parameters
        ----------
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
            Default is 20 epochs.
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement.
            Default is 0.0001.

        Returns
        -------
        self
            Returns self for method chaining
        """
        import matplotlib.pyplot as plt
        from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
        import numpy as np

        # Get training and validation data
        X_train, y_train, _ = self.train_dataset
        X_valid, y_valid, _ = self.valid_dataset

        # Define a custom learning rate scheduler
        def cyclic_learning_rate(epoch, initial_lr=self.learning_rate):
            """
            Cyclical learning rate with warmup and decay

            Parameters:
            -----------
            epoch : int
                Current training epoch
            initial_lr : float
                Initial learning rate

            Returns:
            --------
            float
                Adjusted learning rate for the current epoch
            """
            # Warmup phase (first 5 epochs)
            if epoch < 5:
                return initial_lr * (1 + epoch) / 5

            # Cyclic learning rate
            cycle_epochs = 10  # Length of one learning rate cycle
            cycle = np.floor(1 + epoch / (2 * cycle_epochs))
            x = np.abs(epoch / cycle_epochs - 2 * cycle + 1)

            # Decay the peak learning rate over time
            decay_factor = 0.9 ** (epoch // (2 * cycle_epochs))
            lr = initial_lr * max(0, (1 - x)) * decay_factor

            return max(lr, 1e-6)  # Ensure a minimum learning rate

        # Define callbacks with enhanced learning rate management
        callbacks = [
            # Early stopping with best weights restoration
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                min_delta=min_delta,
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),

            # Adaptive learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,   # Reduce learning rate by half
                patience=3,   # Wait 5 epochs before reducing learning rate
                min_lr=1e-6,  # Lower bound on learning rate
                mode='min',
                verbose=1
            ),

            # Custom learning rate scheduler
            LearningRateScheduler(cyclic_learning_rate, verbose=1)
        ]

        # Train the model
        print(f"\nTraining model for up to {self.epochs} epochs...")
        print(f"Learning Rate Scheduling:")
        print(f"  - Initial learning rate: {self.learning_rate}")
        print(f"  - Warmup epochs: 5")
        print(f"  - Cyclic learning rate with decay")
        print(f"Early stopping:")
        print(f"  - Patience: {patience} epochs")
        print(f"  - Minimum delta: {min_delta}")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Store the training history
        self.history = history.history

        # Plot training history with learning rate
        plt.figure(figsize=(15, 10))

        # Loss subplot
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Metrics subplot
        plt.subplot(2, 2, 2)
        if self.model_type == 'classification':
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
        else:
            plt.plot(history.history['mean_absolute_error'], label='Training MAE')
            plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
            plt.title('Model MAE')
            plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()

        # Learning rate subplot
        plt.subplot(2, 2, 3)
        lr_history = [cyclic_learning_rate(epoch) for epoch in range(len(history.history['loss']))]
        plt.plot(lr_history, label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.yscale('log')  # Use log scale for better visualization

        plt.tight_layout()
        plt.show()

        # Print final training information
        final_epoch = len(history.history['loss'])
        print(f"\nTraining completed in {final_epoch} epochs")

        # Print best validation metrics
        if self.model_type == 'classification':
            best_val_accuracy = max(history.history['val_accuracy'])
            best_val_loss = min(history.history['val_loss'])
            print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
        else:
            best_val_mae = min(history.history['val_mean_absolute_error'])
            best_val_loss = min(history.history['val_loss'])
            print(f"Best Validation MAE: {best_val_mae:.4f}")

        print(f"Best Validation Loss: {best_val_loss:.4f}")

        return self
    def augment_data_method(self, augmentation_factor=2):
        """
        Augment training data using molecular transformations.

        Parameters
        ----------
        augmentation_factor : int
            Number of variants to generate per original molecule

        Returns
        -------
        self
            Returns self for method chaining
        """
        # Validate current dataset
        if self.train_dataset is None:
            raise ValueError("Data must be loaded before augmentation.")

        X_train, y_train, smiles_train = self.train_dataset

        # Convert fingerprints back to molecules
        mols_train = []
        for mol_smiles in smiles_train:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is not None:
                mols_train.append(mol)

        # Perform data augmentation
        print(f"Augmenting training data by factor of {augmentation_factor}...")
        augmented_mols, augmented_labels = MolecularDataAugmentation.generate_synthetic_molecules(
            mols_train, y_train, augmentation_factor
        )

        # Recalculate fingerprints for augmented molecules
        augmented_X = self.descriptor_calculator.calculate_for_mols(augmented_mols)

        # Update training dataset with augmented data
        self.train_dataset = (
            augmented_X,
            augmented_labels,
            [Chem.MolToSmiles(mol) for mol in augmented_mols]
        )

        # Print augmentation statistics
        original_size = len(X_train)
        augmented_size = len(augmented_X)
        print(f"Original training data size: {original_size}")
        print(f"Augmented training data size: {augmented_size}")
        print(f"Augmentation multiplier: {augmented_size / original_size:.2f}")

        # Analyze augmented dataset
        self._analyze_dataset()

        return self

    def evaluate(self, dataset='test'):
        """
        Evaluate the model on the specified dataset.

        Parameters
        ----------
        dataset : str, optional
            Dataset to evaluate on ('train', 'valid', 'test'), by default 'test'

        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        if dataset not in ['train', 'valid', 'test']:
            raise ValueError("Invalid dataset. Choose 'train', 'valid', or 'test'.")

        if dataset == 'train':
            X, y, _ = self.train_dataset
        elif dataset == 'valid':
            X, y, _ = self.valid_dataset
        else:
            X, y, _ = self.test_dataset

        # Make predictions
        y_pred = self.model.predict(X)

        # Ensure consistent shapes
        y = y.reshape(-1)  # Flatten to 1D array
        y_pred = y_pred.reshape(-1)  # Flatten to 1D array

        # Calculate metrics based on model type
        if self.model_type == 'classification':
            # Apply threshold for binary classification
            y_pred_binary = (y_pred > 0.5).astype(int)  # Assuming 0.5 threshold

            metrics = {
                'accuracy': accuracy_score(y, y_pred_binary),
                'precision': precision_score(y, y_pred_binary),
                'recall': recall_score(y, y_pred_binary),
                'f1_score': f1_score(y, y_pred_binary)
            }

        else:  # Regression
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }

        # Print metrics
        print(f"\nModel evaluation on {dataset} set:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        return metrics

    def visualize_model_performance(self, dataset='test'):
        """
        Visualize model performance with various plots.

        Parameters
        ----------
        dataset : str
            Dataset to visualize ('train', 'valid', 'test')

        Returns
        -------
        None
        """
        if self.model is None:
            raise ValueError("Model must be trained before visualizing performance.")

        # Get the specified dataset
        if dataset == 'train':
            X, y, smiles = self.train_dataset
            title_prefix = "Training"
        elif dataset == 'valid':
            X, y, smiles = self.valid_dataset
            title_prefix = "Validation"
        else:  # test
            X, y, smiles = self.test_dataset
            title_prefix = "Test"

        # Generate predictions
        y_pred_prob = self.model.predict(X)

        # For classification models
        if self.model_type == 'classification':
            y_pred = (y_pred_prob > 0.5).astype(int)

            # Create a figure with multiple subplots
            plt.figure(figsize=(20, 15))

            # 1. Confusion Matrix
            plt.subplot(2, 2, 1)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{title_prefix} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            # 2. ROC Curve
            plt.subplot(2, 2, 2)
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{title_prefix} ROC Curve')
            plt.legend(loc='lower right')

            # 3. Prediction Distribution
            plt.subplot(2, 2, 3)
            sns.histplot(y_pred_prob, bins=50, kde=True)
            plt.axvline(0.5, color='red', linestyle='--')
            plt.title(f'{title_prefix} Prediction Distribution')
            plt.xlabel('Prediction Probability')
            plt.ylabel('Count')

            # 4. Precision-Recall Curve
            plt.subplot(2, 2, 4)
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y, y_pred_prob)
            ap = average_precision_score(y, y_pred_prob)
            plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{title_prefix} Precision-Recall Curve')
            plt.legend(loc='lower left')
        else:  # For regression models
            # Create a figure with multiple subplots
            plt.figure(figsize=(20, 15))

            # 1. Actual vs Predicted Plot
            plt.subplot(2, 2, 1)
            plt.scatter(y, y_pred_prob, alpha=0.5)
            min_val = min(np.min(y), np.min(y_pred_prob))
            max_val = max(np.max(y), np.max(y_pred_prob))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.title(f'{title_prefix} Actual vs Predicted')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')

            # 2. Residual Plot
            plt.subplot(2, 2, 2)
            residuals = y - y_pred_prob
            plt.scatter(y_pred_prob, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'{title_prefix} Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')

            # 3. Residual Distribution
            plt.subplot(2, 2, 3)
            sns.histplot(residuals, kde=True)
            plt.title(f'{title_prefix} Residual Distribution')
            plt.xlabel('Residual Value')

            # 4. Prediction Distribution
            plt.subplot(2, 2, 4)
            sns.histplot(y_pred_prob, bins=30, color='blue', label='Predicted', alpha=0.5)
            sns.histplot(y, bins=30, color='red', label='Actual', alpha=0.5)
            plt.title(f'{title_prefix} Value Distribution')
            plt.xlabel('Value')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def save(self, model_path=None):
        """
        Save the trained model and associated information.

        Parameters
        ----------
        model_path : str, optional
            Path to save the model, by default based on dataset name

        Returns
        -------
        str
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving.")

        if model_path is None:
            model_path = f"{self.dataset_name}_model.keras"  # Add .keras extension

        # Save Keras model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        # Optionally save additional information
        import json
        info = {
            "dataset": self.dataset_name,
            "model_type": self.model_type,
            "hidden_units": self.hidden_units,
            "dropout": self.dropout,
            "fp_radius": self.fp_radius,
            "fp_bits": self.fp_bits,
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(f"{model_path}_info.json", "w") as f:
            json.dump(info, f, indent=4)

        print(f"Model metadata saved to {model_path}_info.json")

        return model_path


# Example usage
if __name__ == "__main__":
    # Check if running in Google Colab
    in_colab = check_if_colab()
    if in_colab:
        mount_google_drive()
        file_path = get_file_path_from_drive()
    else:
        file_path = "bace.csv"  # Default local path

    print(f"Using file: {file_path}")

    # Initialize and run the pipeline
    pipeline = DrugDiscoveryPipeline(
        dataset_name="bace_classification",
        split_type="scaffold",
        fp_radius=3,
        fp_bits=2048
    )

    # Load data
    pipeline.load_data(
        compound_file=file_path,
        smiles_column='mol',
        label_column='Class'
    )
    # Augment data before scaling and training
    pipeline.augment_data_method()

    # Scale features
    pipeline.scale_features()

    # Build and train model
    pipeline.build_model(
    model_type='classification',
    hidden_units=(64, 32,),  # More layers with reduced units
    dropout=0.4,  # Slightly increased dropout
    learning_rate=0.001,
    batch_size=40,
    epochs=50,
    regularization_strength=0.001,
    l1_ratio=0.3
    )
    # Example usage with custom early stopping
    pipeline.train(
        patience=5,    # Wait 20 epochs for improvement
        min_delta=0.001  # Smaller threshold for improvement
    )

    # Evaluate on test set
    test_metrics = pipeline.evaluate(dataset='test')

    # Visualize performance
    pipeline.visualize_model_performance(dataset='test')

    # Save the model
    model_path = pipeline.save()
    print(f"Model saved to {model_path}")

    # Make predictions on new compounds function
    def predict_new_compounds(pipeline, smiles_list):
        """
        Make predictions on new compounds.

        Parameters
        ----------
        pipeline : DrugDiscoveryPipeline
            Trained pipeline object
        smiles_list : List[str]
            List of SMILES strings to predict

        Returns
        -------
        np.ndarray
            Predictions for each compound
        """
        # Convert SMILES to molecules
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        valid_mols = [mol for mol in mols if mol is not None]

        if len(valid_mols) < len(smiles_list):
            print(f"Warning: {len(smiles_list) - len(valid_mols)} invalid SMILES strings removed")

        if not valid_mols:
            print("No valid molecules to predict")
            return None

        # Calculate fingerprints
        X = pipeline.descriptor_calculator.calculate_for_mols(valid_mols)

        # Scale features if scaler exists
        if pipeline.feature_scaler is not None:
            X = pipeline.feature_scaler.transform(X)

        # Make predictions
        predictions = pipeline.model.predict(X)

        # Process predictions based on model type
        if pipeline.model_type == 'classification':
            # Return both probabilities and binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            result = {
                'probabilities': predictions.flatten(),
                'predicted_class': binary_predictions.flatten()
            }
        else:
            # For regression, just return the predicted values
            result = {
                'predicted_values': predictions.flatten()
            }

        return result

    # Example using the prediction function
    new_compounds = [
        "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
        "CC(C)(C)OC(=O)NC1CCN(CC1)C(=O)C1=CC=C(C=C1)CC1=CC=CC=C1",
        "COC1=CC(=CC(=C1)OC)C(=O)NCCNC(=O)C1=CC(=C(C=C1)OC)OC"
    ]

    print("\nPredicting new compounds...")
    predictions = predict_new_compounds(pipeline, new_compounds)

    if predictions:
        for i, smiles in enumerate(new_compounds):
            if pipeline.model_type == 'classification':
                prob = predictions['probabilities'][i]
                pred_class = predictions['predicted_class'][i]
                print(f"Compound {i+1}: Probability = {prob:.4f}, Class = {pred_class}")
            else:
                value = predictions['predicted_values'][i]
                print(f"Compound {i+1}: Predicted value = {value:.4f}")

    def visualize_model_architecture(pipeline, filename="model_architecture.png"):
        """
        Generate and save a visualization of the model architecture.
        """
        try:
            from tensorflow.keras.utils import plot_model
            plot_model(
                pipeline.model,
                to_file=filename,
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB"
            )
            print(f"Model architecture saved to {filename}")
        except ImportError as e:
            print(f"Could not visualize model architecture: {e}")

    visualize_model_architecture(pipeline)
    2
    print("\nDrug discovery pipeline complete!")