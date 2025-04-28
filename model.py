## model.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, Dict
import logging
from scipy.linalg import companion


class DiagonalParam(nn.Module):
    """
    Diagonal parameterization: A = diag(alpha), where alpha is a learnable vector.
    """

    def __init__(self, state_dim: int, alpha_range: Tuple[float, float] = (-0.5, 0.5)):
        """
        Args:
            state_dim (int): Dimension n of the state.
            alpha_range (tuple): (low, high) for uniform initialization of alpha.
        """
        super().__init__()
        self.n = state_dim
        self.alpha = nn.Parameter(torch.empty(self.n))
        low, high = alpha_range
        nn.init.uniform_(self.alpha, low, high)

    def matrix(self) -> torch.Tensor:
        """Return A = diag(alpha)."""
        return torch.diag(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A to state x: x @ A^T (since A is diagonal, elementwise multiply).
        Args:
            x (Tensor): shape (..., n)
        Returns:
            Tensor: shape (..., n)
        """
        return x * self.alpha


class CompanionParam(nn.Module):
    """
    Companion parameterization: A is a companion matrix defined by coefficients a.
    """

    def __init__(
        self,
        state_dim: int,
        init_from_true: bool = False,
        true_eigvals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Args:
            state_dim (int): Dimension n of the state.
            init_from_true (bool): If True and true_eigvals provided, initialize from true eigenvalues.
            true_eigvals (array-like or Tensor): True eigenvalues of A*, length n.
        """
        super().__init__()
        self.n = state_dim
        # a holds [a1, a2, ..., an]
        if init_from_true:
            if true_eigvals is None:
                 raise ValueError("CompanionParam requires 'true_eigvals' when 'init_from_true' is True.")
            # Convert to numpy array
            if isinstance(true_eigvals, torch.Tensor):
                ev = true_eigvals.detach().cpu().numpy()
            else:
                ev = np.asarray(true_eigvals)
            # poly coefficients: [1, c1, ..., cn] for λ^n + c1 λ^{n-1} + ... + cn = 0
            coeffs = np.poly(ev)  # length n+1
            # Companion a_i should satisfy last column = -a_n,...,-a_1
            # Here coeffs[1:] = [c1, ..., cn]
            # set a = [a1,...,an] = [-cn, ..., -c1]  reversed
            a_init = -coeffs[1:][::-1]
            self.a = nn.Parameter(torch.from_numpy(a_init.astype(np.float32)))
        else:
            # Initialize with small random noise instead of zeros
            std_dev = 0.01
            initial_a = torch.randn(self.n, dtype=torch.float32) * std_dev
            self.a = nn.Parameter(initial_a)
            logging.info(f"CompanionParam initialized 'a' with randn(std={std_dev})")

    def matrix(self) -> torch.Tensor:
        """Construct and return the companion matrix A of shape (n,n)."""
        device = self.a.device
        dtype = self.a.dtype
        A = torch.zeros((self.n, self.n), device=device, dtype=dtype)
        # Set sub-diagonal entries A[i+1, i] = 1
        idx = torch.arange(self.n - 1, device=device)
        A[idx + 1, idx] = 1.0
        # Set last column: A[0, n-1] = -a_n, ..., A[n-1, n-1] = -a_1
        A[:, self.n - 1] = -self.a.flip(0)
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A to state x: x @ A^T
        Args:
            x (Tensor): shape (..., n)
        Returns:
            Tensor: shape (..., n)
        """
        A = self.matrix()
        return x.matmul(A.t())


class LowRankParam(nn.Module):
    """
    Low-rank plus diagonal parameterization: A = diag(D_vec) + U V^T
    """

    def __init__(
        self,
        state_dim: int,
        rank: int,
        diag_range: Tuple[float, float] = (-0.5, 0.5),
        U_std: float = 0.01,
        V_std: float = 0.01,
    ):
        """
        Args:
            state_dim (int): Dimension n of the state.
            rank (int): Rank r of the low-rank component.
            diag_range (tuple): (low, high) for uniform init of diagonal D.
            U_std (float): Standard deviation for normal init of U.
            V_std (float): Standard deviation for normal init of V.
        """
        super().__init__()
        self.n = state_dim
        self.rank = rank
        # Diagonal component
        self.D_vec = nn.Parameter(torch.empty(self.n))
        low, high = diag_range
        nn.init.uniform_(self.D_vec, low, high)
        # Low-rank factors
        self.U = nn.Parameter(torch.randn(self.n, self.rank) * U_std)
        self.V = nn.Parameter(torch.randn(self.n, self.rank) * V_std)

    def matrix(self) -> torch.Tensor:
        """Return A = diag(D_vec) + U V^T."""
        D = torch.diag(self.D_vec)
        return D + self.U @ self.V.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A to state x: x @ A^T
        Args:
            x (Tensor): shape (..., n)
        Returns:
            Tensor: shape (..., n)
        """
        A = self.matrix()
        return x.matmul(A.t())


class HiPPOParam(nn.Module):
    """
    HiPPO-based parameterization using LegS (Legendre-based) discretization.
    Implements proper HiPPO matrix initialization and discretization via bilinear transform.
    """

    def __init__(self, state_dim: int, hippo_config: Dict):
        """
        Args:
            state_dim (int): Dimension n of the state.
            hippo_config (dict): Configuration for HiPPO initialization parameters.
        """
        super().__init__()
        self.n = state_dim
        
        # Create continuous-time HiPPO matrix (n×n)
        A_c = self._init_hippo_legs(self.n)
        
        # Discretize via bilinear transform (Tustin method)
        # A_d = (I + dt/2 * A_c)(I - dt/2 * A_c)^{-1}
        dt = hippo_config.get('dt', 0.01)  # discretization step size
        
        # Bilinear transform
        I = torch.eye(self.n)
        dt_A_c = dt * A_c / 2
        num = I + dt_A_c
        den = I - dt_A_c
        A_d = torch.linalg.solve(den, num)
        
        # Register as parameter (trainable)
        self.A = nn.Parameter(A_d)

    def _init_hippo_legs(self, n: int) -> torch.Tensor:
        """
        Create LegS basis (Legendre-based) continuous-time HiPPO matrix.
        See HiPPO paper (Gu et al.) for mathematical details.
        
        Args:
            n (int): State dimension
            
        Returns:
            torch.Tensor: continuous-time HiPPO matrix of shape (n, n)
        """
        A = torch.zeros((n, n))
        
        # Fill diagonal and subdiagonal terms
        for k in range(n):
            # Diagonal entries: A[k,k] = -(2k+1)
            A[k, k] = -(2*k + 1)
            
            # Sub-diagonal entries
            if k > 0:
                A[k, k-1] = k
        
        return A

    def matrix(self) -> torch.Tensor:
        """Return the discrete-time HiPPO matrix A."""
        return self.A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A to state x: x @ A^T (identity => returns x).
        Args:
            x (Tensor): shape (..., n)
        Returns:
            Tensor: shape (..., n)
        """
        return x


class Model(nn.Module):
    """
    State-space model that encapsulates A_theta parameterization along with
    B, C, D matrices for input/output mappings.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config (dict): Full configuration dictionary parsed from YAML.
        """
        super().__init__()
        self.config = config

        # Data dimensions
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        
        # Determine state dimension 'n': Prioritize model config, fallback to data config
        if 'state_dim' in model_cfg:
            self.n = int(model_cfg['state_dim'])
            logging.info(f"Using state_dim n={self.n} from model config.")
        elif 'state_dim' in data_cfg:
            self.n = int(data_cfg['state_dim'])
            logging.warning(f"Using state_dim n={self.n} from data config (consider moving to model config).")
        else:
            dims = data_cfg.get('state_dims', [])
            if not dims:
                raise ValueError("No state_dim specified in model config and no state_dim or state_dims found in data config.")
            self.n = int(dims[0]) # Use first data dim as last resort
            logging.warning(f"Using first state_dim n={self.n} from data.state_dims (consider specifying model.state_dim).")
        
        self.m = int(data_cfg.get('input_dim', 1))
        self.p = int(data_cfg.get('output_dim', 1))

        # Model parameterization selection
        # determine param_kind
        if 'parameterization_type' in model_cfg:
            param_kind = model_cfg['parameterization_type']
        else:
            # Added a clear error message if the expected key is missing
            raise ValueError(
                "Configuration error: 'model.parameterization_type' key is missing in the config."
            )

        # Instantiate A_param
        kind = param_kind.strip().lower() # Strip whitespace and convert to lower
        
        # Get the init config section consistently
        init_cfg = model_cfg.get('hippo_config', {}).get('init', {})
        if not init_cfg:
            logging.warning("Could not find 'model.hippo_config.init' section in config. Using defaults where possible.")

        if kind == 'diagonal':
            diag_init_cfg = init_cfg.get('diagonal', {})
            if 'alpha_uniform_range' not in diag_init_cfg:
                 raise ValueError("Config Error: 'model.hippo_config.init.diagonal.alpha_uniform_range' is missing.")
            init_range = tuple(diag_init_cfg['alpha_uniform_range'])
            self.A_param = DiagonalParam(self.n, init_range)

        elif kind == 'companion':
            comp_init_cfg = init_cfg.get('companion', {})
            init_true = bool(comp_init_cfg.get('init_from_true_eigenvalues', False))
            sys_cfg = config.get('system', {}) # Get system config from full config, not model_cfg
            true_eigs = sys_cfg.get('true_eigenvalues', None)
            
            if init_true and true_eigs is None:
                raise ValueError(
                    "Configuration Error: 'init_from_true_eigenvalues' is True in model.hippo_config.init.companion, "
                    "but 'system.true_eigenvalues' was not found in the main config file. "
                    "Please provide the true eigenvalues under the 'system' key."
                )
            self.A_param = CompanionParam(self.n, init_true, true_eigs)

        elif kind in ('lowrank', 'low_rank'):
            lowrank_init_cfg = init_cfg.get('lowrank', {})
            diag_init_cfg = init_cfg.get('diagonal', {})
            
            # Read rank 'r' - Check if it's under model directly OR in init config
            # Prioritize model.low_rank_r if present, else look in init.lowrank.r
            if 'low_rank_r' in model_cfg:
                 rank = int(model_cfg['low_rank_r'])
                 logging.info(f"Using rank r={rank} from model.low_rank_r")
            elif 'r' in lowrank_init_cfg:
                 rank = int(lowrank_init_cfg['r'])
                 logging.warning(f"Using rank r={rank} from model.hippo_config.init.lowrank.r (consider moving to model.low_rank_r)")
            else:
                 raise ValueError("Configuration Error: Rank 'r' is missing. Provide either 'model.low_rank_r' or 'model.hippo_config.init.lowrank.r'.")

            # Check other required lowrank/diagonal init params using the corrected path
            if 'alpha_uniform_range' not in diag_init_cfg:
                 raise ValueError("Configuration Error: 'model.hippo_config.init.diagonal.alpha_uniform_range' is missing, needed by LowRankParam.")
            diag_rng = tuple(diag_init_cfg['alpha_uniform_range'])
            
            if 'U_std' not in lowrank_init_cfg:
                raise ValueError("Configuration Error: 'model.hippo_config.init.lowrank.U_std' is missing.")
            U_std = float(lowrank_init_cfg['U_std'])
            
            if 'V_std' not in lowrank_init_cfg:
                raise ValueError("Configuration Error: 'model.hippo_config.init.lowrank.V_std' is missing.")
            V_std = float(lowrank_init_cfg['V_std'])
            
            self.A_param = LowRankParam(self.n, rank, diag_rng, U_std, V_std)

        elif kind in ('hippo', 'hi_p_p_o', 'hippo-based'):
            # Pass the full hippo_config section to HiPPOParam
            full_hippo_cfg = model_cfg.get('hippo_config', {})
            if not full_hippo_cfg:
                 raise ValueError("Configuration Error: 'model.hippo_config' section is missing for HiPPO parameterization.")
            self.A_param = HiPPOParam(self.n, full_hippo_cfg) 

        else:
            raise ValueError(f"Unknown parameterization kind '{param_kind}'")

        # Input, state, output maps
        # B: u -> state
        self.B = nn.Linear(self.m, self.n, bias=False)
        # C: state -> output
        self.C = nn.Linear(self.n, self.p, bias=False)
        # D: u -> output
        self.D = nn.Linear(self.m, self.p, bias=False)
        
        # --- Explicit Initialization --- #
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.xavier_uniform_(self.C.weight)
        # Note: D initialization might depend on whether it represents feedthrough
        # Default nn.Linear initialization is usually okay for D

        # Spectral norm enforcement flag
        self.enforce_spectral_norm = bool(model_cfg.get('apply_spectral_norm', False)) # Corrected key
        
        # Apply spectral norm if requested
        if self.enforce_spectral_norm:
            # Apply spectral norm to B and C linear modules
            from utils import spectral_norm_proj
            self.B = spectral_norm_proj(self.B)
            self.C = spectral_norm_proj(self.C)

        # Forecasting horizons
        train_cfg = config.get('training', {})
        self.horizons = train_cfg.get('horizons', [1, 10, 100])
        self.h_max = int(train_cfg.get('h_max', max(self.horizons)))

    def _get_A_matrix(self) -> torch.Tensor:
        """Constructs the dense A matrix based on the parameterization."""
        # Use the matrix method from the specific A_param object
        return self.A_param.matrix()

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulates the LTI system forward in time.

        Args:
            u (torch.Tensor): Input sequence, shape (batch_size, T, m).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                y_hat_sequence (torch.Tensor): Output sequence, shape (batch_size, T, p).
                x_sequence (torch.Tensor): State sequence, shape (batch_size, T+1, n). Includes x0.
        """
        batch_size, T, _ = u.shape
        x = torch.zeros(batch_size, self.n, device=u.device) # Initial state x0 = 0
        y_hat_list = []
        x_list = [x] # Store initial state x0

        # Get the effective A matrix for this forward pass
        A_eff = self._get_A_matrix()
        
        # --- Add regularization to A to prevent singularity during backward pass ---
        epsilon = 1e-4
        identity_matrix = torch.eye(self.n, device=A_eff.device, dtype=A_eff.dtype)
        A_eff = A_eff + epsilon * identity_matrix
        # ------------------------------------------------------------------------

        for k in range(T):
            u_k = u[:, k, :] # Shape (batch_size, m)
            # State update: x_{k+1} = A x_k + B u_k
            # Need to handle batch dimension: A is (n,n), x is (bs, n), B is (n,m), u_k is (bs, m)
            # Matmul expects (bs, n, n) @ (bs, n, 1) or (n, n) @ (bs, n).T
            # x_k shape (bs, n) -> transpose for matmul: (bs, n, 1)
            x_k_T = x.unsqueeze(-1) # (bs, n, 1)
            A_eff_batch = A_eff.unsqueeze(0).expand(batch_size, -1, -1) # (bs, n, n)
            Bu_k = self.B(u_k)
            # Correct Bu_k: B (n,m), u_k (bs, m) -> B @ u_k.T -> (n, m) @ (m, bs) -> (n, bs)
            # Need result shape (bs, n)
            Bu_k_correct = Bu_k # (bs, n)
            
            # State update: x = Ax + Bu
            x = (A_eff_batch @ x_k_T).squeeze(-1) + Bu_k_correct # (n,n)@(n,bs)->(n,bs)->(bs,n) + (bs,n)

            # Output equation: y_k = C x_k + D u_k
            y_k_hat = self.C(x) + self.D(u_k) # Shape: (bs, p)
            y_hat_list.append(y_k_hat)
            x_list.append(x) # Store state x_{k+1}

        y_hat_sequence = torch.stack(y_hat_list, dim=1) # Shape (batch_size, T, p)
        x_sequence = torch.stack(x_list, dim=1)       # Shape (batch_size, T+1, n)
        return y_hat_sequence, x_sequence

    def step(self, u_k: torch.Tensor, x_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs one step of the LTI system.

        Args:
            u_k (torch.Tensor): Input at step k, shape (batch_size, m).
            x_k (torch.Tensor): State at step k, shape (batch_size, n).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                y_k_hat (torch.Tensor): Output at step k, shape (batch_size, p).
                x_k_plus_1 (torch.Tensor): State at step k+1, shape (batch_size, n).
        """
        A_eff = self._get_A_matrix()
        
        # State update: x_{k+1} = A x_k + B u_k
        Bu_k = self.B(u_k)
        x_k_plus_1 = (A_eff @ x_k.T).T + Bu_k # (bs, n)
        
        # Output equation: y_k = C x_k + D u_k (using x_k, not x_{k+1})
        y_k_hat = self.C(x_k) + self.D(u_k) # (bs, p)
        
        return y_k_hat, x_k_plus_1


def get_model_A(model: nn.Module) -> Optional[torch.Tensor]:
    """Retrieves the effective A matrix from the model, handling parameterizations."""
    if isinstance(model, nn.DataParallel):
        model = model.module # Get the underlying model

    if hasattr(model, '_get_A_matrix') and callable(model._get_A_matrix):
        try:
            return model._get_A_matrix()
        except Exception as e:
            logging.error(f"Error calling model._get_A_matrix: {e}")
            return None
    elif hasattr(model, 'A') and isinstance(model.A, nn.Parameter):
        # Fallback for simple dense case if _get_A_matrix doesn't exist
        return model.A
    else:
        logging.warning("Could not retrieve A matrix from model. Missing '_get_A_matrix' method or 'A' parameter.")
        return None