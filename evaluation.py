## evaluation.py

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from loss import Loss
from model import Model
from typing import Tuple
import scipy.stats as stats
from scipy import linalg
from spectral_loss import compute_spectral_loss
from model import get_model_A
import os
from visualize import plot_eigenvalues, plot_trajectory
import logging
import numpy as np
import ot # Python Optimal Transport library
import json
from pathlib import Path

class Evaluation:
    """
    Evaluation class for computing forecasting MSE, spectral matching loss,
    and system identification error on a test dataset of synthetic LTI systems.
    
    This implementation uses true 2-Wasserstein distance for spectral evaluation
    instead of the surrogate Hungarian matching loss used during training.

    Methods:
        __init__: sets up model, dataloader, device, and loss module.
        evaluate: runs evaluation loop and returns metrics dict.
        wasserstein_distance: computes the 2-Wasserstein distance between eigenvalue distributions.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        config: Dict[str, Any],
        output_dir: str
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            model (nn.Module): Trained state-space model.
            dataloader (DataLoader): DataLoader for the test split.
            config (dict): Configuration dict (parsed from YAML).
            output_dir (str): Directory to save evaluation artifacts.
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.output_dir = output_dir

        # Infer device from model parameters (or default to CPU)
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # Ensure model is on correct device and in eval mode
        self.model.to(self.device)
        self.model.eval()

        # Loss module for pred and spec losses
        self.loss_module = Loss(config)

        # Evaluation horizons and max horizon
        train_cfg = config.get("training", {})
        self.horizons = list(train_cfg.get("horizons", [1, 10, 100]))
        self.h_max = int(train_cfg.get("h_max", max(self.horizons)))

        self.metrics_to_compute = config.get('evaluation', {}).get('metrics_to_compute', ['loss'])
        self.forecast_horizons = config.get('evaluation', {}).get('forecast_horizons', [1, 10, 100])

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.

        Returns:
            Dict[str, float]: Dictionary with keys
                'eval_forecast_mse': average forecasting MSE,
                'eval_spec_loss':   average spectral loss,
                'eval_id_error':    average identification error.
        """
        # Disable grad and set model to eval
        torch.set_grad_enabled(False)
        self.model.eval()

        total_pred_loss = 0.0
        total_spec_loss = 0.0
        total_id_error = 0.0
        total_samples = 0

        metrics = {m: [] for m in self.metrics_to_compute if m != 'forecast_mse'} # Initialize accumulators
        if 'forecast_mse' in self.metrics_to_compute:
             for h in self.forecast_horizons:
                 metrics[f'forecast_mse_{h}'] = []

        all_hungarian_losses = [] # Store Hungarian spectral losses
        all_wasserstein_distances = [] # Store Wasserstein distances

        plots_generated = False # Flag to generate plots only once

        # Lists to store eigenvalues across all batches/samples for final saving
        all_true_eigs_list = [] 
        all_pred_eigs_list = []

        model_type = self.config.get('model_type', 'ssm').lower()
        logging.info(f"Starting evaluation for model type '{model_type}', saving plots to {self.output_dir}")

        metrics_accum = {m: [] for m in self.metrics_to_compute}
        # Ensure forecast_mse keys are initialized even if 'forecast_mse' isn't explicitly listed but horizons are
        for h in self.forecast_horizons:
             metrics_accum[f'forecast_mse_{h}'] = []
             
        for batch_idx, batch in enumerate(self.dataloader):
            # Unpack batch data
            u = batch["u"].to(self.device)           # (batch, T, m)
            y = batch["y"].to(self.device)           # (batch, T, p)
            eig_true = batch.get("eig_true") # Use .get() as it might not always be present/needed
            if eig_true is not None: eig_true = eig_true.to(self.device) # (batch, n), complex
            A_true = batch.get("A")
            if A_true is not None: A_true = A_true.to(self.device) # (batch, n, n)
            
            batch_size, T, _ = y.shape

            # Forward rollout - Unpack the tuple
            y_pred_seq, _ = self.model(u)  # (batch, T, p)

            # Get learned state matrix A_pred and its eigenvalues eig_pred
            A_pred = None
            eig_pred = None
            eig_pred_np = None
            can_compute_eigs = False
            logging.debug(f"Current model type: {model_type}") # Moved log earlier

            # Use the helper function to get A for ANY relevant model type
            if model_type == 'lti_ssm' or model_type == 'randomssm': # Transformer doesn't have A
                try:
                    logging.debug("Attempting to retrieve A_pred using self.get_model_A(self.model)...")
                    A_pred = self.get_model_A(self.model) # <<< Use the helper function
                    
                    if A_pred is not None:
                        logging.debug(f"Successfully retrieved A_pred with shape: {A_pred.shape}")
                        # Compute eigenvalues from A_pred
                        logging.debug("Attempting to compute eigenvalues using torch.linalg.eigvals(A_pred)...")
                        eig_pred = torch.linalg.eigvals(A_pred) # Shape (n,), complex
                        # Check for NaNs in eigenvalues before converting to numpy
                        if not torch.isnan(eig_pred).any():
                             eig_pred_np = eig_pred.detach().cpu().numpy()
                             can_compute_eigs = True
                             logging.debug("Successfully computed eigenvalues.")
                        else:
                             logging.warning("Eigenvalue computation resulted in NaNs.")
                             can_compute_eigs = False
                    else:
                         # get_model_A already logs a warning if it returns None
                         logging.warning(f"get_model_A returned None for model type {model_type}. Cannot compute spectral metrics.")
                         can_compute_eigs = False
                         
                except Exception as e:
                    logging.error(f"Error occurred during A_pred retrieval via get_model_A or eigenvalue computation: {e}", exc_info=True)
                    can_compute_eigs = False # Disable further attempts
            elif model_type == 'simpletransformer':
                 logging.debug("Skipping A_pred retrieval for SimpleTransformer as it lacks a state matrix A.")
                 can_compute_eigs = False # Explicitly set for transformer
            
            # Further ensure metrics are skipped if eigs failed
            if not can_compute_eigs:
                 # Remove spectral metrics if eigenvalues couldn't be computed
                 metrics_to_remove = ['wasserstein_distance', 'hungarian_spectral_loss', 'id_error_norm'] # Add others if needed
                 for metric_key in metrics_to_remove:
                     if metric_key in self.metrics_to_compute:
                          self.metrics_to_compute.remove(metric_key)
                     if metric_key in metrics_accum:
                          del metrics_accum[metric_key]
                 if any(m in ['wasserstein_distance', 'hungarian_spectral_loss', 'id_error_norm'] for m in self.metrics_to_compute): # Check if removal happened
                     logging.warning("Cannot compute eigenvalues, removing relevant spectral metrics from computed metrics.")

            # --- Generate Plots (First Batch, First Item) --- 
            if not plots_generated and batch_idx == 0 and batch_size > 0:
                logging.info("Generating evaluation plots for the first sample...")
                try:
                    # Eigenvalue plot (only if SSM and eigenvalues available)
                    if can_compute_eigs and eig_pred_np is not None and eig_true is not None:
                        true_eigs_sample = eig_true[0].detach().cpu().numpy()
                        eig_plot_path = os.path.join(self.output_dir, 'evaluation_eigenvalues.png')
                        plot_eigenvalues(true_eigs_sample, eig_pred_np, eig_plot_path, 
                                         title=f'Eigenvalues ({model_type.upper()} - Seed: {self.config.get("seed", "N/A")})')
                        logging.info(f"Saved eigenvalue plot to {eig_plot_path}")
                    elif model_type == 'ssm' or model_type == 'lti_ssm' or model_type == 'simpletransformer' or model_type == 'randomssm': # Log only if it was expected
                         logging.warning("Skipping eigenvalue plot as true or learned eigenvalues could not be computed.")
                    
                    # Trajectory plot (applicable to all models)
                    true_y_sample = y[0] # (T, p)
                    # Use the unpacked y_pred_seq here as well
                    pred_y_sample = y_pred_seq[0] # (T, p)
                    traj_plot_path = os.path.join(self.output_dir, 'evaluation_trajectory_output0.png')
                    plot_trajectory(true_y_sample, pred_y_sample, traj_plot_path, 
                                    title=f'Sample Trajectory ({model_type.upper()} - Output 0, Seed: {self.config.get("seed", "N/A")})',
                                    output_dim_to_plot=0)
                    logging.info(f"Saved trajectory plot to {traj_plot_path}")
                    
                    plots_generated = True # Set flag
                except Exception as e:
                    logging.error(f"Failed to generate plots: {e}", exc_info=True)
                    plots_generated = True # Avoid retrying if error occurs

            # --- Compute Batch Metrics (Conditional on Available Data) --- #
            
            # Initialize batch losses to zero
            batch_spec_loss = 0.0
            batch_id_error = 0.0
            
            # Spectral Metrics (Wasserstein) and Eigenvalue Saving
            if can_compute_eigs and eig_true is not None:
                # Append eigenvalues FOR THIS BATCH to lists
                true_eigs_np = eig_true.detach().cpu().numpy() # (batch, n)
                all_true_eigs_list.append(true_eigs_np)
                pred_eigs_repeated = np.repeat(eig_pred_np[np.newaxis, :], batch_size, axis=0) # (batch, n)
                all_pred_eigs_list.append(pred_eigs_repeated)

                # Compute spectral loss (Wasserstein distance) for the batch
                current_batch_spec_loss = 0.0
                for b in range(batch_size):
                    # Use the already computed numpy arrays
                    true_eigs = true_eigs_np[b] 
                    pred_eigs = eig_pred_np # This is (n,), same for all batch items
                    w2_dist = self.wasserstein_distance(true_eigs, pred_eigs)
                    current_batch_spec_loss += w2_dist
                
                batch_spec_loss = current_batch_spec_loss # Assign to batch_spec_loss
                total_spec_loss += batch_spec_loss # Accumulate total
                
            elif model_type == 'ssm' or model_type == 'lti_ssm' or model_type == 'simpletransformer' or model_type == 'randomssm': # Log warning only if spectral metrics were expected
                 logging.warning("Skipping eigenvalue saving and spectral loss calculation for batch due to missing true or predicted eigenvalues.")

            # Forecasting MSE loss (always computed)
            # Build multi-horizon predictions and targets
            T_eff = T - self.h_max
            if T_eff <= 0:
                raise ValueError(f"Effective time T_eff must be > 0, got {T_eff}")

            # y_pred_h, y_true_h: (batch, T_eff, H, p)
            y_pred_h = torch.stack(
                [y_pred_seq[:, t : t + T_eff, :]
                for t in range(self.h_max + 1)], dim=2 # Check indices carefully
            ) # Incorrect slicing logic here previously
            # Let's assume horizons define start indices relative to T_eff steps
            y_pred_h_correct = torch.stack(
                [y_pred_seq[:, h : h + T_eff, :]
                for h in range(1, self.h_max + 1)], dim=2 
            ) # Shape: (batch, T_eff, h_max, p)
            y_true_h_correct = torch.stack(
                [y[:, h : h + T_eff, :]
                for h in range(1, self.h_max + 1)], dim=2
            ) # Shape: (batch, T_eff, h_max, p)
            
            # Select only the required horizons for loss calculation
            horizon_indices = [h-1 for h in self.horizons] # 0-indexed
            y_pred_h_final = y_pred_h_correct[:, :, horizon_indices, :]
            y_true_h_final = y_true_h_correct[:, :, horizon_indices, :]

            L_pred = self.loss_module.pred_loss(y_pred_h_final, y_true_h_final)
            total_pred_loss += L_pred.item() * batch_size

            # System ID Error
            if model_type == 'lti_ssm' and A_pred is not None and A_true is not None:
                current_batch_id_error = 0.0
                for b in range(batch_size):
                    a_pred_sample = A_pred.detach().cpu().numpy() # (n, n)
                    a_true_sample = A_true[b].detach().cpu().numpy() # (n, n)
                    id_err = self._compute_id_error(a_pred_sample, a_true_sample)
                    current_batch_id_error += id_err
                batch_id_error = current_batch_id_error
                total_id_error += batch_id_error # Accumulate total
            elif model_type == 'lti_ssm': # Log warning only if ID error was expected
                logging.warning("Skipping ID error calculation for batch due to missing A_pred or A_true.")

            # Forecasting MSE (per horizon)
            # Check if any forecast MSE metric is requested
            if any(f'forecast_mse_{h}' in metrics_accum for h in self.forecast_horizons):
                batch_forecast_mses = self._compute_forecast_mse(y, y_pred_seq, self.forecast_horizons)
                for key, mse_val in batch_forecast_mses.items():
                    if key in metrics_accum:
                        # Append the average MSE for this horizon over the batch
                        metrics_accum[key].append(mse_val) 
                    
            # Hungarian Spectral Loss (Non-Differentiable)
            if 'hungarian_spectral_loss' in metrics_accum and (model_type == 'lti_ssm' or model_type == 'simpletransformer' or model_type == 'randomssm') and A_pred is not None and A_true is not None:
                try:
                    # Ensure A_true is batched (B, N, N)
                    if A_true.dim() == 2: # If true A is somehow not batched
                        logging.warning("A_true seems to be 2D, attempting to unsqueeze for batch Hungarian loss computation.")
                        A_true_batch = A_true.unsqueeze(0).repeat(batch_size, 1, 1)
                    elif A_true.dim() == 3 and A_true.shape[0] == batch_size:
                        A_true_batch = A_true # Already batched
                    else:
                         raise ValueError(f"Unexpected shape for A_true: {A_true.shape}. Expected ({batch_size}, n, n) or (n, n).")
                    
                    # Call the imported function
                    # Pass single A_pred (n, n) and batched A_true_batch (bs, n, n)
                    # loss_type='hungarian_l2_non_diff' computes non-differentiable loss on CPU
                    batch_loss_tensor = compute_spectral_loss(
                        A_theta=A_pred.detach(), # Use detached A_pred
                        A_star_batch=A_true_batch.detach(), # Use detached A_true_batch
                        loss_type='hungarian_l2_non_diff'
                    )
                    batch_loss_val = batch_loss_tensor.item() # Get scalar value
                    
                    # Check if loss is valid before appending
                    if not np.isnan(batch_loss_val) and not np.isinf(batch_loss_val):
                       metrics_accum['hungarian_spectral_loss'].append(batch_loss_val)
                    else:
                       logging.warning(f"compute_spectral_loss (hungarian) returned invalid value ({batch_loss_val}) for batch.")

                except Exception as e:
                    # Log specific error from compute_spectral_loss if it occurs
                    logging.warning(f"compute_spectral_loss (hungarian) failed for the batch: {e}", exc_info=False) 
            elif (model_type == 'lti_ssm' or model_type == 'simpletransformer' or model_type == 'randomssm') and 'hungarian_spectral_loss' in metrics_accum:
                 # Log warning if loss was expected but A matrices were missing
                 if A_pred is None or A_true is None: # Only log if matrices were missing
                     logging.warning("Skipping Hungarian loss calculation for batch due to missing A_pred or A_true.")

            total_samples += batch_size
            # Log progress (optional)
            if (batch_idx + 1) % 10 == 0:
                logging.debug(f"Evaluation Batch [{batch_idx+1}/{len(self.dataloader)}] Processed")
        
        # --- End of Batch Loop ---

        # --- Finalize Metrics --- #
        final_metrics = {}
        for key, values in metrics_accum.items():
            if values: # Check if list is not empty
                final_metrics[key] = np.mean(values)
            else:
                # Avoid logging warning if the metric was expected to be empty (e.g., W2 dist for non-SSM)
                should_warn = True
                if key == 'wasserstein_distance' and model_type != 'lti_ssm': should_warn = False
                if key == 'id_error_norm' and model_type != 'lti_ssm': should_warn = False
                if key == 'hungarian_spectral_loss' and model_type != 'lti_ssm': should_warn = False
                
                if should_warn:
                    logging.warning(f"Metric '{key}' had no values recorded during evaluation.")
                final_metrics[key] = float('nan') 

        # --- Save Accumulated Eigenvalues --- #
        if all_true_eigs_list and all_pred_eigs_list:
            try:
                # Concatenate eigenvalues from all batches
                final_true_eigs = np.concatenate(all_true_eigs_list, axis=0)
                final_pred_eigs = np.concatenate(all_pred_eigs_list, axis=0)
                
                eigenvalues_path = os.path.join(self.output_dir, 'eigenvalues.npz')
                np.savez(eigenvalues_path, true=final_true_eigs, pred=final_pred_eigs)
                logging.info(f"Saved accumulated true and predicted eigenvalues to {eigenvalues_path}")
            except Exception as e:
                logging.error(f"Failed to save accumulated eigenvalues: {e}", exc_info=True)
        elif model_type == 'lti_ssm': # Only warn if expected for SSM
            logging.warning("No eigenvalues were accumulated, skipping saving eigenvalues.npz.")

        # --- Save Final Metrics --- #
        try:
            # Ensure output directory is a Path object and exists
            output_dir_path = Path(self.output_dir).resolve() # Get absolute path
            logging.info(f"Attempting to save metrics. Output directory: {output_dir_path}")
            #logging.info(f"Does output directory exist? {output_dir_path.is_dir()}")
            output_dir_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
            #logging.info(f"Output directory existence ensured.")

            metrics_path = output_dir_path / 'metrics.json'
            logging.info(f"Metrics file full path: {metrics_path}")
            
            # Convert numpy types to standard Python types for JSON serialization
            serializable_metrics = {k: (v.item() if hasattr(v, 'item') else v) 
                                    for k, v in final_metrics.items() if not np.isnan(v)} # Filter out NaNs
            
            logging.info(f"Attempting to open {metrics_path} for writing...")
            with open(metrics_path, 'w') as f:
                #logging.info(f"File {metrics_path} opened successfully. Dumping JSON...")
                json.dump(serializable_metrics, f, indent=4)
                #logging.info(f"JSON dumped successfully.")
            logging.info(f"Saved final evaluation metrics to {metrics_path}")
        except Exception as e:
            logging.error(f"Failed to save metrics.json: {e}", exc_info=True)

        logging.info(f"Evaluation finished. Final averaged metrics: {final_metrics}")
        return final_metrics

    def get_model_A(self, model: torch.nn.Module) -> torch.Tensor:
        """Helper to get the state matrix A from the model."""
        target_model = model
        # Handle DataParallel wrapper first
        if isinstance(model, torch.nn.DataParallel):
            target_model = model.module

        # Check if it's our custom Model class
        if isinstance(target_model, Model):
            try:
                # Use the internal method to get the matrix
                A = target_model._get_A_matrix()
                return A
            except Exception as e:
                logging.error(f"Error calling _get_A_matrix on Model instance: {e}", exc_info=True)
                return None

        # --- Keep checks for other potential structures (Baselines?) ---
        elif hasattr(target_model, 'ssm') and hasattr(target_model.ssm, 'A'): # Common structure in some libraries
            return target_model.ssm.A
        elif hasattr(target_model, 'A'): # Direct attribute (might be used by RandomSSM or others)
            return target_model.A
        elif hasattr(target_model, 'dynamics') and hasattr(target_model.dynamics, 'A'):
             return target_model.dynamics.A
        elif hasattr(target_model, 'backbone') and hasattr(target_model.backbone, 'A'): # If SSM is within a backbone
             return target_model.backbone.A
        # Add specific checks for baseline types if they store A differently
        elif isinstance(target_model, RandomSSM) and hasattr(target_model, 'A'):
             return target_model.A # Assuming RandomSSM stores it directly as A
        elif isinstance(target_model, SimpleTransformer):
             # Transformer likely doesn't have a single 'A' matrix in the LTI sense
             logging.debug("SimpleTransformer does not have a state matrix 'A' attribute.")
             return None 

        logging.warning("Could not automatically find state matrix 'A' in the model structure.")
        return None

    def wasserstein_distance(self, eig_true: np.ndarray, eig_pred: np.ndarray) -> float:
        """
        Computes the 2-Wasserstein distance between two sets of complex eigenvalues.
        
        Eigenvalues are treated as points in the 2D complex plane (real, imaginary).
        Uses the Python Optimal Transport (POT) library.

        Args:
            eig_true (np.ndarray): Ground truth eigenvalues (n,). Complex dtype.
            eig_pred (np.ndarray): Predicted eigenvalues (n,). Complex dtype.

        Returns:
            float: The computed 2-Wasserstein distance. Returns np.inf on error.
        """
        if eig_true.shape != eig_pred.shape:
            logging.error(f"Shape mismatch in Wasserstein distance: {eig_true.shape} vs {eig_pred.shape}")
            return np.inf # Return infinity for shape mismatch
        
        if len(eig_true) == 0:
            return 0.0 # Distance is 0 if both sets are empty

        # Convert complex eigenvalues to 2D points (real, imag)
        # POT expects arrays of shape (n_samples, n_features=2)
        points_true = np.stack([eig_true.real, eig_true.imag], axis=-1)
        points_pred = np.stack([eig_pred.real, eig_pred.imag], axis=-1)
        
        # Ensure inputs are float64 for POT stability
        points_true = points_true.astype(np.float64)
        points_pred = points_pred.astype(np.float64)

        # Uniform weights for both distributions
        n = len(eig_true)
        a = np.ones((n,)) / n
        b = np.ones((n,)) / n

        # Compute the pairwise squared Euclidean distance matrix between points
        # M_ij = ||points_true[i] - points_pred[j]||^2
        M = ot.dist(points_true, points_pred, metric='sqeuclidean')
        
        # Ensure M is C-contiguous and float64
        M = np.ascontiguousarray(M, dtype=np.float64)

        try:
            # Compute the exact 2-Wasserstein distance (squared) using EMD
            # ot.emd2 returns the squared distance
            w2_squared = ot.emd2(a, b, M) 
        
            # Check for potential negative values due to numerical issues
            if w2_squared < 0:
             logging.warning(f"Negative squared Wasserstein distance encountered ({w2_squared}). Clamping to zero.")
             w2_squared = 0.0
             
            # Return the actual Wasserstein distance (sqrt of the result)
            return np.sqrt(w2_squared)

        except Exception as e:
            logging.error(f"POT Wasserstein distance calculation failed: {e}")
            # Fallback or error signaling
            # Maybe try 1D approach as fallback? Or return Inf?
            # Returning Inf to signal a computation error clearly.
            return np.inf 

    def _compute_forecast_mse(self, y_true_seq, y_pred_seq, horizons) -> Dict[str, float]:
        """
        Computes the Mean Squared Error (MSE) for specified forecast horizons.
        
        Args:
            y_true_seq (Tensor): True sequences (batch, T, p).
            y_pred_seq (Tensor): Predicted sequences (batch, T, p).
            horizons (list[int]): List of forecast horizons (e.g., [1, 10, 100]).
            
        Returns:
            Dict[str, float]: Dictionary mapping horizon keys (e.g., 'forecast_mse_1') to MSE values.
        """
        batch_size, T, p = y_true_seq.shape
        forecast_mses = {}
        h_max = max(horizons) if horizons else 0
        
        # Calculate effective sequence length for multi-step predictions
        T_eff = T - h_max
        if T_eff <= 0:
            logging.warning(f"Sequence length T={T} is too short for max horizon h_max={h_max}. Cannot compute multi-step MSE.")
            for h in horizons:
                forecast_mses[f'forecast_mse_{h}'] = float('nan')
            return forecast_mses
            
        # Prepare tensors for multi-horizon MSE calculation
        # y_true_h will be (batch, T_eff, num_horizons, p)
        # y_pred_h will be (batch, T_eff, num_horizons, p)
        horizon_indices = [h - 1 for h in horizons] # 0-based indices for slicing
        
        # Extract true values at future horizons
        y_true_h = torch.stack(
            [y_true_seq[:, h : h + T_eff, :]
            for h in horizons
        ], dim=2)
        
        # Extract predictions made *now* for future horizons
        y_pred_h = torch.stack(
            [y_pred_seq[:, h : h + T_eff, :]
            for h in horizons
        ], dim=2)
        
        # Calculate MSE for each requested horizon
        for i, h in enumerate(horizons):
            mse = torch.mean((y_true_h[:, :, i, :] - y_pred_h[:, :, i, :])**2)
            forecast_mses[f'forecast_mse_{h}'] = mse.item()
            
        return forecast_mses

    def _compute_id_error(self, A_pred: np.ndarray, A_true: np.ndarray) -> float:
        """
        Computes ||A_pred - A_true||_F / ||A_true||_F
        """
        if A_pred is None or A_true is None:
            return float('nan')
        try:
            with torch.no_grad():
                diff_norm = torch.norm(torch.tensor(A_pred, dtype=torch.float32) - torch.tensor(A_true, dtype=torch.float32), p='fro')
                true_norm = torch.norm(torch.tensor(A_true, dtype=torch.float32), p='fro')
                if true_norm < 1e-9: # Avoid division by zero
                    return float('inf') if diff_norm > 1e-9 else 0.0
                id_error = diff_norm / true_norm
            return id_error.item()
        except Exception as e:
            print(f"Error computing ID error: {e}")
            return float('nan')

def identification_error(params_true: Dict[str, np.ndarray], params_pred: Dict[str, np.ndarray]) -> float:
    pass
