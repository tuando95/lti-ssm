## dataset_loader.py

import random
from typing import List, Dict, Union
import logging
import tqdm # Add tqdm import

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import qr, inv
import control # Added for Lyapunov equation

from utils import set_seed, controllability_check, observability_check


class SyntheticSystem:
    """
    Encapsulates a single synthetic LTI system with known matrices and trajectory.

    Attributes:
        A (np.ndarray): State transition matrix of shape (n, n).
        B (np.ndarray): Input matrix of shape (n, m).
        C (np.ndarray): Output matrix of shape (p, n).
        D (np.ndarray): Feedthrough matrix of shape (p, m).
        u (np.ndarray): Input sequence of shape (T, m).
        y (np.ndarray): Output sequence of shape (T, p).
        eig_true (np.ndarray): Array of true eigenvalues of A, shape (n,).
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.u = u
        self.y = y
        # Compute true eigenvalues once
        self.eig_true = np.linalg.eigvals(self.A)

    @property
    def state_dim(self) -> int:
        """Return the state dimensionality n."""
        return self.A.shape[0]

    @property
    def input_dim(self) -> int:
        """Return the input dimensionality m."""
        return self.B.shape[1]

    @property
    def output_dim(self) -> int:
        """Return the output dimensionality p."""
        return self.C.shape[0]


class SyntheticSystemDataset(Dataset):
    """
    PyTorch Dataset wrapper around a list of SyntheticSystem instances.
    Each item is a dict containing:
      'u'        : torch.FloatTensor of shape (T, m)
      'y'        : torch.FloatTensor of shape (T, p)
      'eig_true' : torch.ComplexFloatTensor of shape (n,)
      'A'        : torch.FloatTensor of shape (n, n)  # Ground Truth A
      'B'        : torch.FloatTensor of shape (n, m)  # Ground Truth B
      'C'        : torch.FloatTensor of shape (p, n)  # Ground Truth C
      # 'D'      : torch.FloatTensor of shape (p, m) # Optional Ground Truth D
    """

    def __init__(self, systems: List[SyntheticSystem]) -> None:
        super().__init__()
        self.systems = systems

    def __len__(self) -> int:
        return len(self.systems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        system = self.systems[idx]
        # Convert numpy arrays stored in SyntheticSystem to torch tensors
        item = {
            'u': torch.from_numpy(system.u).float(),
            'y': torch.from_numpy(system.y).float(),
            'eig_true': torch.from_numpy(system.eig_true).cfloat(), # Use complex float
            'A': torch.from_numpy(system.A).float(),
            'B': torch.from_numpy(system.B).float(),
            'C': torch.from_numpy(system.C).float(),
            # Optionally include D if needed for evaluation
            # 'D': torch.from_numpy(system.D).float()
        }
        return item


class DatasetLoader:
    """
    Generates synthetic LTI systems with controlled spectra, splits into
    train/val/test, and provides corresponding DataLoaders.

    Methods:
        generate_systems(): populates self.train_systems, self.val_systems,
                            self.test_systems lists.
        get_dataloader(split): returns a PyTorch DataLoader for the given split.
    """

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config (dict): Configuration dictionary with keys 'training' and 'data'.
        """
        self.config = config
        # Ensure reproducibility
        seed = int(self.config.get('seed', 42))
        set_seed(seed)

        # Placeholders for system lists
        self.train_systems: List[SyntheticSystem] = []
        self.val_systems: List[SyntheticSystem] = []
        self.test_systems: List[SyntheticSystem] = []

        # Generate and split systems
        self.generate_systems()

    def generate_systems(self) -> None:
        """
        Generate synthetic LTI systems across spectral classes and state dims,
        ensuring controllability & observability, and simulate trajectories.
        Splits each class+dim into train/val/test subsets.
        """
        print("start generating systems")
        data_cfg = self.config.get('data', {})
        train_cfg = self.config.get('training', {})
        model_cfg = self.config.get('model', {}) # Get model config

        # --- Determine Target State Dimension for Filtering ---
        if 'state_dim' in model_cfg:
            target_n = int(model_cfg['state_dim'])
            logging.info(f"DatasetLoader: Will filter systems for target state_dim n={target_n}")
        else:
            # Fallback or error if not specified? For now, let's raise an error if filtering is needed but target isn't set.
            state_dims_in_data = data_cfg.get('state_dims', [])
            if len(state_dims_in_data) > 1:
                raise ValueError("Dataset config specifies multiple state_dims, but model.state_dim is missing. Cannot determine which dimension to filter for.")
            elif len(state_dims_in_data) == 1:
                target_n = int(state_dims_in_data[0])
                logging.warning(f"Model state_dim not specified, defaulting to the single data dim: {target_n}")
            else:
                 raise ValueError("No state dimensions found in data.state_dims or specified in model.state_dim.")
        # ----------------------------------------------------

        T = int(train_cfg.get('sequence_length', 10000))
        sigma = float(data_cfg.get('noise_sigma', 1e-4))
        m = int(data_cfg.get('input_dim', 1))
        p = int(data_cfg.get('output_dim', 1))
        state_dims = data_cfg.get('state_dims', [8, 16])
        spectral_classes = data_cfg.get('spectral_classes', [])
        systems_per = int(data_cfg.get('systems_per_class_and_dim', 100))
        split = data_cfg.get('split', [0.8, 0.1, 0.1])
        validate_autocorr = bool(data_cfg.get('validate_autocorrelation', False)) # Default to False
        autocorr_max_lag = int(data_cfg.get('autocorrelation_max_lag', 5))       
        autocorr_rtol = float(data_cfg.get('autocorrelation_rtol', 0.2)) # Relaxed default tolerance

        # Normalize split ratios
        total_ratio = sum(split)
        if not np.isclose(total_ratio, 1.0):
            split = [s / total_ratio for s in split]
        train_ratio, val_ratio, test_ratio = split

        all_generated_systems = [] # Collect systems of ALL dimensions first

        for n in state_dims:
            for spec_class in spectral_classes:
                group = [] # Reinitialize group for each (n, spec_class) combination
                pbar_desc = f"Generating systems (n={n}, class={spec_class})"
                for i in tqdm.tqdm(range(systems_per), desc=pbar_desc, leave=False):
                    try:
                        logging.debug(f"System {i+1}/{systems_per}: Starting generation.")
                        # 1) Sample eigenvalues
                        lambdas = self._sample_eigenvalues(n, spec_class)
                        logging.debug(f"System {i+1}: Eigenvalues sampled.")
                        # 2) Build A = V diag(lambdas) V^{-1}
                        A = self._generate_A_with_eigenvalue_constraints(lambdas)
                        logging.debug(f"System {i+1}: Matrix A generated.")
                        # 3) Sample B, C until controllable & observable
                        logging.debug(f"System {i+1}: Finding controllable/observable B, C...")
                        cont_obs_attempts = 0
                        max_cont_obs_attempts = 20000 # Increased from 1000
                        found_B_C = False # Flag to track if B, C were found for this A
                        while cont_obs_attempts < max_cont_obs_attempts:
                            cont_obs_attempts += 1
                            B = np.random.randn(n, m) * (1.0 / np.sqrt(n))
                            C = np.random.randn(p, n) * (1.0 / np.sqrt(n))
                            
                            # Use PBH test - requires eigenvalues (lambdas)
                            controllable = controllability_check(A, B, lambdas)
                            observable = observability_check(A, C, lambdas)

                            if controllable and observable:
                                found_B_C = True
                                break # Found B, C, exit C/O loop
                        
                        # Check if C/O check succeeded for this A
                        if found_B_C:
                            logging.debug(f"System {i+1}: Found valid B, C after {cont_obs_attempts} attempts.")
                            # 4) Feedthrough
                            D = np.zeros((p, m), dtype=float)

                            # 5) Simulate trajectory
                            logging.debug(f"System {i+1}: Simulating trajectory...")
                            u, y = self._simulate_trajectory(A, B, C, D)
                            logging.debug(f"System {i+1}: Trajectory simulated.")

                            # Optional: Autocorrelation Validation (if needed, add back here)
                            autocorr_valid = True # Placeholder if not validating
                            if validate_autocorr:
                                # ... (autocorrelation validation code) ...
                                pass # Keep existing code here

                            if autocorr_valid: # Only add if valid (or validation skipped)
                                system_data = SyntheticSystem(A, B, C, D, u, y)
                                all_generated_systems.append(system_data) # Add system to the combined list
                                logging.debug(f"System {i+1}: Added to collection (dim={n}, class={spec_class})")
                        else:
                            logging.warning(f"System {i+1}: Failed to find controllable/observable B, C after {max_cont_obs_attempts} attempts. Skipping. (dim={n}, class={spec_class})")

                    except Exception as e:
                        logging.error(f"Error generating system {i+1} (n={n}, class={spec_class}): {e}", exc_info=True)

        # --- Filter Systems by Target Dimension --- #
        original_count = len(all_generated_systems)
        # Use the state_dim property for filtering
        filtered_systems = [sys for sys in all_generated_systems if sys.state_dim == target_n]
        filtered_count = len(filtered_systems)
        logging.info(f"Filtered systems: Kept {filtered_count} out of {original_count} total generated systems matching target n={target_n}.")
        if filtered_count == 0:
            logging.error(f"CRITICAL: No systems generated or kept matching target dimension n={target_n}. Check config and generation process.")
            # Depending on desired behavior, could raise error or just proceed with empty lists.
        # ------------------------------------------ #

        # --- Split the FILTERED Systems --- #
        num_systems = len(filtered_systems)
        num_train = int(np.floor(train_ratio * num_systems))
        num_val = int(np.floor(val_ratio * num_systems))
        # Ensure reproducibility of the shuffle
        np.random.shuffle(filtered_systems)
        self.train_systems = filtered_systems[:num_train]
        self.val_systems = filtered_systems[num_train : num_train + num_val]
        self.test_systems = filtered_systems[num_train + num_val :]
        # ---------------------------------- #

        logging.info(
            f"Systems generated and split - Train: {len(self.train_systems)}, "
            f"Val: {len(self.val_systems)}, Test: {len(self.test_systems)}"
        )
        print("end generating systems")

    def _sample_eigenvalues(self, n: int, spec_class: str) -> np.ndarray:
        """
        Sample an array of n eigenvalues according to the specified spectral class.

        Args:
            n (int): Number of eigenvalues.
            spec_class (str): One of the configured spectral categories.

        Returns:
            np.ndarray: Array of complex eigenvalues of length n.
        """
        if spec_class == 'uniform_decay':
            vals = np.random.uniform(0.90, 0.99, size=n)
            result = vals.astype(complex)
            return result

        elif spec_class == 'clustered':
            n1 = n // 2
            n2 = n - n1
            cluster1 = np.random.uniform(0.95 - 0.01, 0.95 + 0.01, size=n1)
            cluster2 = np.random.uniform(0.80 - 0.02, 0.80 + 0.02, size=n2)
            return np.concatenate([cluster1, cluster2]).astype(complex)

        elif spec_class == 'mixed_real_complex':
            real_count = n // 2
            real_parts = np.random.uniform(0.80, 0.98, size=real_count)
            complex_count = n - real_count
            pair_count = complex_count // 2
            radii = np.random.uniform(0.90, 0.99, size=pair_count) # Changed from (0.99, 1.01)
            angles = np.random.uniform(0, 2 * np.pi, size=pair_count)
            pairs = []
            for r, th in zip(radii, angles):
                lam = r * np.exp(1j * th)
                pairs.extend([lam, np.conj(lam)])
            return np.concatenate([real_parts.astype(complex), np.array(pairs)])

        elif spec_class == 'oscillatory':
            angles = np.random.uniform(0, 2 * np.pi, size=n)
            return np.exp(1j * angles)

        elif spec_class == 'hierarchical':
            vals = np.array([0.99 ** ((i + 1) / n) for i in range(n)], dtype=float)
            return vals.astype(complex)

        else:
            raise ValueError(f"Unknown spectral class '{spec_class}'")

    def _generate_A_with_eigenvalue_constraints(self, lambdas: np.ndarray) -> np.ndarray:
        """Construct a real matrix A via similarity transform on a real block-diagonal
        matrix whose eigenvalues are the specified lambdas.
        Ensures A is real and has the desired eigenvalues.
        """
        n = len(lambdas)
        tol = 1e-9 # Tolerance for checking if imaginary part is zero

        # Separate real and complex eigenvalues (ensure complex come in pairs)
        real_eigenvalues = []
        complex_eigenvalues = []
        processed_indices = set()

        for i in range(n):
            if i in processed_indices:
                continue
            lam = lambdas[i]
            if abs(np.imag(lam)) < tol: # Treat as real
                real_eigenvalues.append(np.real(lam))
                processed_indices.add(i)
            else:
                # Find conjugate pair
                found_conjugate = False
                for j in range(i + 1, n):
                    if j in processed_indices:
                        continue
                    if np.isclose(lambdas[j], np.conj(lam), rtol=tol, atol=tol):
                        complex_eigenvalues.append(lam) # Store one of the pair
                        processed_indices.add(i)
                        processed_indices.add(j)
                        found_conjugate = True
                        break
                if not found_conjugate:
                    # This shouldn't happen if _sample_eigenvalues works correctly
                    logging.warning(f"Complex eigenvalue {lam} does not have a conjugate pair. Treating as real.")
                    real_eigenvalues.append(np.real(lam))
                    processed_indices.add(i)

        # Construct the real block-diagonal matrix Lambda_block
        Lambda_block = np.zeros((n, n))
        current_idx = 0

        # Add 1x1 blocks for real eigenvalues
        for lam_r in real_eigenvalues:
            if current_idx < n:
                Lambda_block[current_idx, current_idx] = lam_r
                current_idx += 1
            else:
                logging.error("Error: Too many eigenvalues for matrix size during block construction.")
                break
        
        # Add 2x2 blocks for complex conjugate pairs
        for lam_c in complex_eigenvalues:
            if current_idx + 1 < n:
                a = np.real(lam_c)
                b = np.imag(lam_c)
                Lambda_block[current_idx, current_idx] = a
                Lambda_block[current_idx, current_idx + 1] = b
                Lambda_block[current_idx + 1, current_idx] = -b
                Lambda_block[current_idx + 1, current_idx + 1] = a
                current_idx += 2
            else:
                logging.error("Error: Not enough space for 2x2 block during construction.")
                # Handle potentially remaining single eigenvalue if n was odd and last was complex? Unlikely with sampling logic.
                if current_idx < n: # Place real part if space allows
                    Lambda_block[current_idx, current_idx] = np.real(lam_c)
                    current_idx += 1
                break
        
        if current_idx != n:
             logging.error(f"Mismatch in block diagonal construction: Processed {current_idx} dimensions, expected {n}.")
             # Handle error? Return identity? Raise exception?
             # For now, proceed but log error.

        # Apply random orthogonal similarity transform
        Z = np.random.randn(n, n)
        V, _ = np.linalg.qr(Z)
        A = V @ Lambda_block @ V.T # V is real orthogonal, V.T = V^{-1}

        # Verify eigenvalues (optional, for debugging)
        # eigvals_A = np.linalg.eigvals(A)
        # logging.debug(f"Original lambdas (sorted): {np.sort(lambdas)}")
        # logging.debug(f"Resulting A eigenvalues (sorted): {np.sort(eigvals_A)}")
        
        return A

    def _simulate_trajectory(self, A, B, C, D):
        """Simulates a trajectory for a given LTI system."""
        # Retrieve necessary parameters from config
        data_cfg = self.config.get('data', {})
        train_cfg = self.config.get('training', {})
        T = int(train_cfg.get('sequence_length', 10000))
        sigma = float(data_cfg.get('noise_sigma', 1e-4))
        m = int(data_cfg.get('input_dim', 1))
        p = int(data_cfg.get('output_dim', 1))
        n = A.shape[0]

        # Initialize
        u = np.random.randn(T, m)  # Zero-mean, unit-variance white noise input
        x = np.zeros((T + 1, n))
        y = np.zeros((T, p))
        x[0] = np.zeros(n) # Initial state set to zero

        # Simulate
        for k in range(T):
            x[k+1] = A @ x[k] + B @ u[k]
            y[k]   = C @ x[k] + D @ u[k] + np.random.randn(p) * sigma # Add noise
        
        return u, y

    def _calculate_theoretical_autocovariance(self, A, B, C, D, max_lag: int) -> np.ndarray:
        """
        Calculates the theoretical output autocovariance R_y(tau) = E[y(k+tau) y(k)^T]
           for lags tau = 0 to max_lag. Assumes zero-mean white noise input u ~ N(0, I).
           Assumes D=0 for simplicity for tau > 0.
        """
        n = A.shape[0]
        p = C.shape[0]
        
        # Ensure numpy arrays
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        
        autocov = np.zeros((max_lag + 1, p, p))
        
        try:
            # Solve discrete Lyapunov equation for state covariance P: P = A P A^T + B B^T
            P = control.dlyap(A, B @ B.T)
            
            # R_y(0) = C P C^T + D D^T (Assuming input covariance is Identity)
            autocov[0] = C @ P @ C.T + D @ D.T 
            
            # R_y(tau) = C A^tau P C^T for tau > 0 (Assuming D=0)
            # If D is non-zero, more terms involving input autocorrelation are needed.
            # We'll proceed assuming D=0 for lags > 0, which is common.
            if np.any(D): 
                print("_calculate_theoretical_autocovariance assumes D=0 for lags > 0.")
                
            A_pow_tau = np.identity(n)
            for tau in range(1, max_lag + 1):
                A_pow_tau = A_pow_tau @ A # A^tau
                autocov[tau] = C @ A_pow_tau @ P @ C.T
                
        except Exception as e:
            print(f"Failed to solve Lyapunov equation or compute autocovariance: {e}")
            # Return NaNs or zeros if calculation fails
            autocov.fill(np.nan)
            
        return autocov

    def _calculate_empirical_autocorrelation(self, y: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculates the empirical output autocorrelation r_y(tau) = sum_k[y(k+tau) y(k)] / T
           for lags tau = 0 to max_lag.
           Returns only the diagonal elements (autocorrelation of each output dim).
        """
        y = np.asarray(y) # Ensure numpy array (T, p)
        T, p = y.shape
        autocorr = np.zeros((max_lag + 1, p))

        if T <= max_lag:
            print(f"Sequence length T={T} is not greater than max_lag={max_lag}. Cannot compute empirical autocorrelation reliably.")
            autocorr.fill(np.nan)
            return autocorr
            
        for i in range(p):
            y_dim = y[:, i]
            # Use numpy.correlate in 'full' mode. The result length is 2*T - 1.
            # The correlation at lag tau is at index (T - 1 + tau).
            np_corr = np.correlate(y_dim, y_dim, mode='full') 
            # Normalize by sequence length T
            np_corr /= T
            # Extract lags 0 to max_lag
            autocorr[:, i] = np_corr[T - 1 : T + max_lag] # Indices T-1 (lag 0) to T-1+max_lag
            
        return autocorr

    def get_dataloader(self, split: str) -> DataLoader:
        """
        Create a PyTorch DataLoader for the requested split.

        Args:
            split (str): One of 'train', 'val', or 'test'.

        Returns:
            DataLoader: Yields batches of dicts from SyntheticSystemDataset.
        """
        if split not in ('train', 'val', 'test'):
            raise ValueError("split must be one of 'train', 'val', 'test'")

        systems = getattr(self, f"{split}_systems")
        # Check if the systems list for the split is empty
        if not systems:
            logging.warning(f"DatasetLoader: No systems found for split '{split}'. Returning DataLoader with 0 samples.")
            # Create an empty dataset to avoid DataLoader error, but log a warning
            dataset = SyntheticSystemDataset([]) 
        else:
            dataset = SyntheticSystemDataset(systems)

        data_cfg = self.config.get('data', {})
        batch_size = int(data_cfg.get('batch_size_systems', 1))
        num_workers = int(data_cfg.get('num_workers', 0))
        shuffle = split == 'train'
        pin_memory = torch.cuda.is_available()

        # Important: If dataset is empty, DataLoader should still be created but will yield nothing.
        # The original error happens inside RandomSampler if shuffle=True and dataset is empty.
        # We can prevent this by setting shuffle=False if the dataset is empty.
        if not systems:
            shuffle = False 

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
