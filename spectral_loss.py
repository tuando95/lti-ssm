"""
Utility for computing a spectral loss based on Sinkhorn divergence of eigenvalues.

This module provides a differentiable loss function that measures the 
difference between the eigenvalues of a learned state matrix (A_theta) 
and a batch of ground truth state matrices (A_star_batch).

The loss is calculated for each item in the batch by:
1. Computing eigenvalues for A_theta and the corresponding A_star.
2. Converting complex eigenvalues to 2D real vectors.
3. Using the Sinkhorn algorithm (geomloss.SamplesLoss) on the 
   eigenvalue distributions to find the optimal transport plan.
4. Calculating the Sinkhorn divergence for the optimal plan using PyTorch 
   tensors to maintain differentiability w.r.t. A_theta's eigenvalues.
5. Averaging the loss across the batch.

Requires: geomloss (`pip install geomloss`)
"""
import torch
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    import geomloss
    GEOMOSS_AVAILABLE = True
except ImportError:
    GEOMOSS_AVAILABLE = False
    logging.warning("geomloss not found. Differentiable Sinkhorn spectral loss is unavailable.")

logger = logging.getLogger(__name__)


def _compute_eigenvalues(A: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    """Helper to compute eigenvalues, handling potential errors."""
    device = A.device
    dtype = A.dtype
    try:
        # Ensure float or complex for eigvals
        A_comp = A if A.is_complex() else A.to(torch.complex64)
        if requires_grad:
            lambda_vals = torch.linalg.eigvals(A_comp) # Shape (n,)
        else:
            with torch.no_grad():
                lambda_vals = torch.linalg.eigvals(A_comp) # Shape (n,)
    except Exception as e:
        logger.error(f"Eigenvalue computation failed for matrix on device {device}: {e}. Returning NaNs.")
        # Return NaNs in the expected shape and type
        n = A.shape[0]
        lambda_vals = torch.full((n,), float('nan'), device=device, dtype=torch.complex64)
        # If requires_grad, attach a dummy operation to allow potential backprop through the error path
        # This might not be ideal, but prevents crashes. Consider raising error instead.
        if requires_grad:
             lambda_vals = lambda_vals * (A.sum() * 0 + 1) 
    return lambda_vals

def compute_spectral_loss(A_theta: torch.Tensor, A_star_batch: torch.Tensor, 
                          loss_type: str = 'sinkhorn',
                          sinkhorn_blur: float = 0.05, sinkhorn_scaling: float = 0.5, 
                          sinkhorn_debias: bool = True) -> torch.Tensor:
    """
    Computes a spectral loss between a single learned matrix A_theta
    and a batch of ground truth matrices A_star_batch.

    Supports:
    - 'sinkhorn': Differentiable Sinkhorn divergence via geomloss (default, requires geomloss).
    - 'hungarian_l2_non_diff': Non-differentiable Hungarian assignment loss (L2 squared cost).

    Args:
        A_theta (torch.Tensor): Learned state matrix (n x n).
        A_star_batch (torch.Tensor): Batch of ground truth state matrices (bs x n x n).
        loss_type (str): Type of loss ('sinkhorn' or 'hungarian_l2_non_diff').
        sinkhorn_blur (float): Sinkhorn regularization strength (epsilon).
        sinkhorn_scaling (float): Parameter for Sinkhorn algorithm computation.
        sinkhorn_debias (bool): Whether to use debiased Sinkhorn divergence.

    Returns:
        torch.Tensor: Scalar spectral loss value.
                      If 'sinkhorn', it's differentiable and on A_theta's device.
                      If 'hungarian', it's non-differentiable on CPU.
    """
    if A_theta.ndim != 2:
        raise ValueError(f"A_theta must be a 2D matrix (n x n), but got shape {A_theta.shape}")
    if A_star_batch.ndim != 3:
        raise ValueError(f"A_star_batch must be a 3D tensor (bs x n x n), but got shape {A_star_batch.shape}")
    if A_theta.shape[0] != A_theta.shape[1] or A_theta.shape != A_star_batch.shape[1:]:
        raise ValueError(f"A_theta ({A_theta.shape}) and A_star matrices ({A_star_batch.shape[1:]}) must be square and have the same size (n x n).")

    n = A_theta.shape[0]
    bs = A_star_batch.shape[0]
    device = A_theta.device
    dtype = A_theta.dtype 

    # --- Compute Eigenvalues --- 
    # Compute eigenvalues for A_theta (gradients depend on loss_type)
    lambda_theta = _compute_eigenvalues(A_theta, requires_grad=(loss_type == 'sinkhorn'))

    # Compute eigenvalues for A_star_batch (no gradients needed)
    lambda_star_list = []
    for i in range(bs):
        lambda_star_list.append(_compute_eigenvalues(A_star_batch[i], requires_grad=False))
    lambda_star_batch_eig = torch.stack(lambda_star_list) # Shape (bs, n)

    # Check for any NaN eigenvalues (computation errors)
    valid_theta_mask = ~torch.isnan(lambda_theta).any()
    valid_star_mask = ~torch.isnan(lambda_star_batch_eig).any(dim=1) # Mask for successful computations per batch item

    if not valid_theta_mask:
        logger.error(f"Could not compute eigenvalues for A_theta. Returning high loss/NaN based on type.")
        return torch.tensor(1e6 if loss_type == 'sinkhorn' else float('nan'), device=device if loss_type == 'sinkhorn' else 'cpu', dtype=dtype if loss_type=='sinkhorn' else torch.float32)

    if not valid_star_mask.any():
        logger.warning("Spectral loss could not be computed for any A_star item in the batch.")
        if loss_type == 'sinkhorn':
            final_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if A_theta.requires_grad:
                 final_loss = final_loss * (A_theta.sum() * 0 + 1) # Dummy op for grad
            return final_loss
        else: # Hungarian
            return torch.tensor(float('nan'), device='cpu', dtype=torch.float32)
            
    # Filter out failed A_star computations for loss calculation
    lambda_star_batch_eig_valid = lambda_star_batch_eig[valid_star_mask]
    bs_valid = lambda_star_batch_eig_valid.shape[0]

    # --- Compute Loss based on Type --- 
    if loss_type == 'hungarian_l2_non_diff':
        # --- Hungarian Assignment Loss (Non-Differentiable) --- 
        lambda_theta_np = lambda_theta.detach().cpu().numpy()
        lambda_star_batch_np = lambda_star_batch_eig_valid.detach().cpu().numpy()
        
        batch_hungarian_losses = []
        for i in range(bs_valid):
            l_star = lambda_star_batch_np[i] # (n,) complex
            l_theta = lambda_theta_np      # (n,) complex
            
            # Calculate cost matrix: squared Euclidean distance in complex plane
            cost_matrix = np.zeros((n, n), dtype=np.float64)
            for r in range(n):
                for c in range(n):
                    diff = l_theta[r] - l_star[c]
                    cost_matrix[r, c] = diff.real**2 + diff.imag**2
            
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                min_cost = cost_matrix[row_ind, col_ind].sum()
                batch_hungarian_losses.append(min_cost / n) # Average cost per eigenvalue
            except Exception as e:
                logger.warning(f"Hungarian assignment failed for item {i}: {e}")
                batch_hungarian_losses.append(np.nan)

        # Average loss over valid batch items
        final_loss_val = np.nanmean(batch_hungarian_losses) if batch_hungarian_losses else np.nan
        return torch.tensor(final_loss_val, device='cpu', dtype=torch.float32)

    elif loss_type == 'sinkhorn':
        # --- Sinkhorn Divergence (Differentiable) --- 
        if not GEOMOSS_AVAILABLE:
            raise ImportError("geomloss is required for Sinkhorn spectral loss, but it's not installed.")
            
        # Convert Complex Eigenvalues to Real 2D Vectors 
        lambda_theta_real = torch.view_as_real(lambda_theta) # Shape (n, 2)
        lambda_star_batch_real = torch.view_as_real(lambda_star_batch_eig_valid) # Shape (bs_valid, n, 2)

        # Expand lambda_theta to match batch size for geomloss
        lambda_theta_batch_real = lambda_theta_real.unsqueeze(0).expand(bs_valid, n, 2) # (bs_valid, n, 2)

        # Define Sinkhorn loss function (using SamplesLoss for empirical distributions)
        # Treats each set of eigenvalues as a point cloud in 2D (real/imaginary plane)
        # Uses squared Euclidean distance (p=2) as the ground cost between eigenvalues.
        sinkhorn_loss_fn = geomloss.SamplesLoss(
            loss="sinkhorn", 
            p=2,              # Squared L2 ground cost
            blur=sinkhorn_blur,
            scaling=sinkhorn_scaling,
            debias=sinkhorn_debias,
            backend="tensorized" # Suitable for point clouds
        )

        # Compute batched loss. Input shapes: (Batch, N_points, Dimension)
        # Here: (bs_valid, n, 2) for both lambda_theta and lambda_star
        sinkhorn_losses = sinkhorn_loss_fn(lambda_theta_batch_real, lambda_star_batch_real) # Shape (bs_valid,)

        # Average the loss over the valid batch items
        final_loss = sinkhorn_losses.mean()
        return final_loss
        
    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Must be 'sinkhorn' or 'hungarian_l2_non_diff'.")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    n = 4
    bs = 3 # Batch size for A_star
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example A_theta (requires grad)
    A_theta = torch.randn(n, n, dtype=torch.float32, device=device, requires_grad=True)

    # Example A_star_batch (no grad)
    A_star_list = []
    for _ in range(bs):
        # Create matrices with somewhat similar eigenvalues for testing
        eigvals_star = torch.randn(n, dtype=torch.complex64) * 0.8 # Stable-ish
        eigvals_star = torch.sort(eigvals_star, key=lambda x: x.real)[0]
        D = torch.diag(eigvals_star)
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.complex64))
        A_star_i = (Q @ D @ Q.H).real # Create a real matrix
        A_star_list.append(A_star_i)
        
    A_star_batch = torch.stack(A_star_list).to(device=device)

    print("A_theta requires grad:", A_theta.requires_grad)
    print("A_star_batch requires grad:", A_star_batch.requires_grad)

    # Compute loss
    loss = compute_spectral_loss(A_theta, A_star_batch)

    print(f"Computed Spectral Loss (Sinkhorn): {loss.item()}")

    # Test backward pass
    if loss.requires_grad:
        try:
            loss.backward()
            print("Backward pass successful.")
            if A_theta.grad is not None:
                print("Gradient computed for A_theta:")
                # print(A_theta.grad)
                print(f"  Gradient norm: {torch.linalg.norm(A_theta.grad)}")
            else:
                print("Gradient for A_theta is None (check computation graph).")
        except Exception as e:
            print(f"Backward pass failed: {e}")
    else:
        print("Loss does not require gradients.")
