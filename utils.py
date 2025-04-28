"""
utils.py

Utility functions for reproducibility, spectral‐norm enforcement, and
controllability/observability checks in LTI system generation and SSM modeling.
"""

import os
import random
from typing import Any

import numpy as np
import torch
from scipy.linalg import expm
from numpy.linalg import matrix_power
from torch import nn
from torch.nn.utils import spectral_norm
import control
import logging

def set_seed(seed: int) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU and CUDA) to ensure
    reproducibility. Also configures PyTorch CuDNN to be deterministic.

    Args:
        seed (int): The seed to set across all RNGs.
    """
    # Python built‐in RNG
    random.seed(seed)
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy RNG
    np.random.seed(seed)
    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def spectral_norm_proj(module: nn.Module) -> nn.Module:
    """
    Apply spectral normalization to the 'weight' parameter of the given module
    so that its spectral norm is constrained to be ≤ 1. This modifies the module
    in‐place (registers a spectral‐norm hook) and returns it for convenience.

    Args:
        module (nn.Module): A torch module with a 'weight' parameter to normalize.

    Returns:
        nn.Module: The same module instance with spectral normalization applied.

    Raises:
        ValueError: If the module does not have a 'weight' attribute.
    """
    if not hasattr(module, "weight"):
        raise ValueError(
            f"Cannot apply spectral_norm_proj: module of type {type(module)} "
            "has no 'weight' parameter."
        )
    # Wrap the module with spectral normalization
    return spectral_norm(module)


def controllability_check(
    A: np.ndarray, B: np.ndarray, lambdas: np.ndarray, tol: float = 1e-6
) -> bool:
    """
    Check the controllability of the LTI system defined by (A, B) using the PBH test.
    The system is controllable iff rank([lambda * I - A, B]) == n for all eigenvalues lambda.

    Args:
        A (np.ndarray): State transition matrix of shape (n, n).
        B (np.ndarray): Input matrix of shape (n, m).
        lambdas (np.ndarray): Array of eigenvalues of A (n,).
        tol (float): Tolerance for rank computation.

    Returns:
        bool: True if the system is controllable, False otherwise.

    Raises:
        ValueError: If A is not square or B/lambdas have incompatible dimensions.
    """
    A_arr = np.asarray(A, dtype=float)
    B_arr = np.asarray(B, dtype=float)
    lambdas_arr = np.asarray(lambdas, dtype=complex)

    # Validate shapes
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError(f"A must be square. Got shape {A_arr.shape}.")
    n = A_arr.shape[0]
    if B_arr.ndim != 2 or B_arr.shape[0] != n:
        raise ValueError(
            f"B must have shape (n, m) where n={n}. Got shape {B_arr.shape}."
        )
    if lambdas_arr.shape != (n,):
        raise ValueError(f"lambdas must have shape (n,) where n={n}. Got shape {lambdas_arr.shape}")

    # PBH Test: Check rank for each unique eigenvalue
    # Use a slightly looser tolerance for finding unique eigenvalues
    unique_lambdas = np.unique(np.round(lambdas_arr, decimals=8))
    I = np.identity(n)

    for lam in unique_lambdas:
        # Construct the PBH matrix [lambda*I - A, B]
        # Ensure subtraction happens correctly with complex lambda
        pbh_matrix = np.hstack(((lam * I - A_arr), B_arr)).astype(complex)
        rank = np.linalg.matrix_rank(pbh_matrix, tol)
        if rank < n:
            logging.debug(f"PBH Controllability check failed for lambda={lam}: rank={rank}, n={n}")
            # Optional: Log condition number if rank check fails
            # try:
            #     cond_num = np.linalg.cond(pbh_matrix)
            #     logging.debug(f"PBH Matrix Condition number: {cond_num}")
            # except np.linalg.LinAlgError:
            #     logging.debug(f"PBH Matrix Condition number calculation failed.")
            return False

    return True # Passed for all unique eigenvalues


def observability_check(
    A: np.ndarray, C: np.ndarray, lambdas: np.ndarray, tol: float = 1e-6
) -> bool:
    """
    Check the observability of the LTI system defined by (A, C) using the PBH test.
    The system is observable iff rank([lambda * I - A^T, C^T]) == n for all eigenvalues lambda.
    Equivalently, rank([lambda*I - A; C]) == n.

    Args:
        A (np.ndarray): State transition matrix of shape (n, n).
        C (np.ndarray): Output matrix of shape (p, n).
        lambdas (np.ndarray): Array of eigenvalues of A (n,).
        tol (float): Tolerance for rank computation.

    Returns:
        bool: True if the system is observable, False otherwise.

    Raises:
        ValueError: If A is not square or C/lambdas have incompatible dimensions.
    """
    A_arr = np.asarray(A, dtype=float)
    C_arr = np.asarray(C, dtype=float)
    lambdas_arr = np.asarray(lambdas, dtype=complex)

    # Validate shapes
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError(f"A must be square. Got shape {A_arr.shape}.")
    n = A_arr.shape[0]
    if C_arr.ndim != 2 or C_arr.shape[1] != n:
        raise ValueError(
            f"C must have shape (p, n) where n={n}. Got shape {C_arr.shape}."
        )
    if lambdas_arr.shape != (n,):
        raise ValueError(f"lambdas must have shape (n,) where n={n}. Got shape {lambdas_arr.shape}")

    # PBH Test: Check rank for each unique eigenvalue
    # Use a slightly looser tolerance for finding unique eigenvalues
    unique_lambdas = np.unique(np.round(lambdas_arr, decimals=8))
    I = np.identity(n)

    for lam in unique_lambdas:
        # Construct the PBH matrix [lambda*I - A; C]
        # Ensure subtraction happens correctly with complex lambda
        pbh_matrix = np.vstack(((lam * I - A_arr), C_arr)).astype(complex)
        rank = np.linalg.matrix_rank(pbh_matrix, tol)
        if rank < n:
            logging.debug(f"PBH Observability check failed for lambda={lam}: rank={rank}, n={n}")
            # Optional: Log condition number if rank check fails
            # try:
            #     cond_num = np.linalg.cond(pbh_matrix)
            #     logging.debug(f"PBH Matrix Condition number: {cond_num}")
            # except np.linalg.LinAlgError:
            #     logging.debug(f"PBH Matrix Condition number calculation failed.")
            return False

    return True # Passed for all unique eigenvalues

# --- Model Related ---
