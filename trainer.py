## trainer.py

import math
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler as AmpGradScaler
from tqdm import tqdm
import torch.nn.functional as F

from utils import set_seed
from loss import Loss
from spectral_loss import compute_spectral_loss


class Trainer:
    """
    Trainer orchestrates the mixed-precision AdamW training loop with
    linear warmup + cosine decay scheduling, gradient clipping, and
    early stopping based on validation loss improvements.
    
    This trainer follows the paper's methodology of training a separate SSM
    for each individual system rather than a single model across multiple systems.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        output_dir: Path,
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The instantiated state-space model.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            config (dict): Configuration dictionary (parsed from YAML).
            output_dir (Path): Directory to save outputs like checkpoints and loss logs.
        """
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reproducibility
        seed = int(self.config.get("seed", 42))
        set_seed(seed)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Store the passed model instance
        self.model = model.to(self.device) 
        self.config = config # Already assigned, but ensure config is stored

        # Training and validation data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss module
        self.loss_fn = Loss(config)

        # --------------------
        # Optimizer settings
        # --------------------
        opt_cfg = self.config.get("optimizer", {})
        opt_type = opt_cfg.get("type", "AdamW").lower()
        lr = float(opt_cfg.get("learning_rate", 1e-3))
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0) or 0.0)
        self.max_grad_norm = float(opt_cfg.get("gradient_clipping_max_norm", 1.0))

        if opt_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), # Now uses the stored model instance
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {opt_type}")

        # --------------------
        # Scheduler: linear warmup + cosine decay
        # --------------------
        sched_cfg = self.config.get("scheduler", {})
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        total_epochs = int(sched_cfg.get("total_epochs", 20))
        steps_per_epoch = len(self.train_loader)
        total_steps = max(1, total_epochs * steps_per_epoch)
        warmup_steps = max(0, warmup_epochs * steps_per_epoch)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # --------------------
        # Mixed precision
        # --------------------
        self.use_amp = config.get('training', {}).get('use_amp', False)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = AmpGradScaler(device='cuda')
            logging.info("Using Automatic Mixed Precision (AMP) with GradScaler.")
        else:
            self.scaler = AmpGradScaler(enabled=False)
            if self.use_amp and self.device.type != 'cuda':
                logging.warning("AMP requested but device is not CUDA. AMP disabled.")
            else:
                logging.info("Automatic Mixed Precision (AMP) is disabled.")

        # --------------------
        # Early stopping
        # --------------------
        train_cfg = self.config.get("training", {})
        self.early_stop_patience = int(train_cfg.get("early_stop_patience", 10))
        self.best_val_loss = float("inf")
        self.no_improve_count = 0

        # --------------------
        # Objective balance and horizons
        # --------------------
        self.horizons = list(train_cfg.get("horizons", [1, 10, 100]))
        self.h_max = int(train_cfg.get("h_max", max(self.horizons)) if self.horizons else 0) # Handle empty horizons

        # --------------------
        # Checkpoint paths
        # --------------------
        ckpt_cfg = self.config.get("checkpoint", {})
        self.best_ckpt_path = ckpt_cfg.get("best_model_path", "best_checkpoint.pt")
        self.final_ckpt_path = ckpt_cfg.get("final_model_path", "final_checkpoint.pt")

        # Store for training loop
        self.total_epochs = total_epochs

        # Spectral Loss configuration
        self.spectral_loss_weight = float(
            self.config.get("training", {}).get("spectral_loss_weight", 0.0)
        )
        logging.info(
            f"Prediction Loss functions initialized from loss.py. Spectral loss weight: {self.spectral_loss_weight}"
        )

        logging.info(f"Trainer initialized:")
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Optimizer: {opt_type}, LR={lr}, WD={weight_decay}")
        logging.info(f"  Epochs: {total_epochs}, Warmup: {warmup_epochs}")
        logging.info(f"  Horizons: {self.horizons}, h_max: {self.h_max}")

        self.loss_df_path = self.output_dir / "losses.csv"

    def train(self) -> tuple[nn.Module, float]:
        """
        Run the training process by training a single model instance on batches from the dataset.
        
        Returns:
            Tuple[nn.Module, float]: The trained model instance and the best validation loss achieved.
        """
        logging.info("Starting training process...")
        
        # Use self.model directly
        model = self.model

        # Optimizer and scheduler were already initialized with self.model.parameters()
        optimizer = self.optimizer
        
        best_val_loss = float("inf")
        no_improve_count = 0
        best_model_state = None
        best_epoch = 0
        
        loss_data = []
        
        # Initialize loss log file
        if self.loss_df_path.exists():
            logging.warning(f"Loss file {self.loss_df_path} already exists, overwriting.")
            self.loss_df_path.unlink()
        pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_pred_loss', 'train_spec_loss', 
            'val_loss', 'val_pred_loss', 'val_spec_loss', 'lr'
        ]).to_csv(self.loss_df_path, index=False)

        # Outer epoch loop with tqdm
        epochs_pbar = tqdm(range(1, self.total_epochs + 1), desc="Epochs")
        for epoch in epochs_pbar:
            model.train()
            train_loss_accum = 0.0
            train_pred_loss_accum = 0.0
            train_spec_loss_accum = 0.0
            num_train_batches = len(self.train_loader)

            # Inner batch loop with tqdm
            batch_pbar = tqdm(enumerate(self.train_loader), total=num_train_batches, desc=f"Epoch {epoch} Training", leave=False)
            for batch_idx, batch in batch_pbar:
                optimizer.zero_grad()

                with autocast(device_type='cuda', enabled=True):
                    u = batch["u"].to(self.device)
                    y = batch["y"].to(self.device)
                    A_star = batch.get("A", None)
                    if A_star is not None:
                        A_star = A_star.to(self.device)
                    
                    # Unpack the tuple returned by the model
                    y_pred_seq, _ = model(u) # Assuming x_sequence is not needed here
                    
                    y_true_seq = y 
                    # Use the unpacked y_pred_seq for loss
                    L_pred = F.mse_loss(y_pred_seq, y_true_seq)
                    train_pred_loss_accum += L_pred.item()

                    L_spec = torch.tensor(0.0, device=self.device)
                    A_theta = None
                    if hasattr(model, 'A_param') and hasattr(model.A_param, 'matrix'):
                        A_theta = model.A_param.matrix()
                    else:
                        logging.warning("Cannot retrieve A_theta via model.A_param.matrix().")
                    
                    if self.spectral_loss_weight > 0 and A_theta is not None and A_star is not None:
                        try:
                            L_spec = compute_spectral_loss(A_theta, A_star)
                            train_spec_loss_accum += L_spec.item()
                        except Exception as e:
                            logging.error(f"Error computing spectral loss: {e}")
                            L_spec = torch.tensor(0.0, device=self.device)
                    else:
                        logging.warning("Could not retrieve A_theta from model for spectral loss calculation.")
                    
                    loss = L_pred + self.spectral_loss_weight * L_spec
                    train_loss_accum += loss.item()

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                
                self.scheduler.step()
                
                # Update tqdm postfix with running averages
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{train_loss_accum / (batch_idx + 1):.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

            avg_train_loss = train_loss_accum / num_train_batches
            avg_train_pred_loss = train_pred_loss_accum / num_train_batches
            avg_train_spec_loss = train_spec_loss_accum / num_train_batches
            
            val_loss = self.validate(epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                best_model_state = model.state_dict()
                best_epoch = epoch
            else:
                no_improve_count += 1
                if no_improve_count >= self.early_stop_patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
            
            # --- Store losses for CSV --- #
            epoch_loss_data = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_pred_loss': avg_train_pred_loss,
                'train_spec_loss': avg_train_spec_loss,
                'val_loss': val_loss,
                'val_pred_loss': self._validate(model, epoch)['pred_loss'],
                'val_spec_loss': self._validate(model, epoch)['spec_loss'],
                'lr': optimizer.param_groups[0]['lr']
            }
            loss_data.append(epoch_loss_data)
            # Append to CSV incrementally
            pd.DataFrame([epoch_loss_data]).to_csv(self.loss_df_path, mode='a', header=False, index=False)

            # Update epoch pbar postfix
            epochs_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        logging.info("Training finished.")

        if best_model_state:
            logging.info(f"Loading best model state from epoch {best_epoch} with validation loss {best_val_loss:.4f}")
            model.load_state_dict(best_model_state)
        else: # Training finished without early stopping
            logging.info(f"Training completed after {self.total_epochs} epochs. Using final model state.")

        logging.info(f"Training finished. Best validation loss: {best_val_loss:.4f}")
        return model, best_val_loss

    def _train_epoch(self, model, optimizer, scheduler, scaler, epoch, total_epochs):
        model.train()
        train_loss_accum = 0.0
        train_pred_loss_accum = 0.0
        train_spec_loss_accum = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=True):
                u = batch["u"].to(self.device)
                y = batch["y"].to(self.device)
                A_star = batch.get("A", None)
                if A_star is not None:
                    A_star = A_star.to(self.device)
                
                y_pred_seq = model(u)
                
                y_true_seq = y 
                L_pred = F.mse_loss(y_pred_seq, y_true_seq)
                train_pred_loss_accum += L_pred.item()

                L_spec = torch.tensor(0.0, device=self.device)
                A_theta = None
                if hasattr(model, 'A_param') and hasattr(model.A_param, 'matrix'):
                    A_theta = model.A_param.matrix()
                else:
                    logging.warning("Cannot retrieve A_theta via model.A_param.matrix().")
                
                if self.spectral_loss_weight > 0 and A_theta is not None and A_star is not None:
                    try:
                        L_spec = compute_spectral_loss(A_theta, A_star)
                        train_spec_loss_accum += L_spec.item()
                    except Exception as e:
                        logging.error(f"Error computing spectral loss: {e}")
                        L_spec = torch.tensor(0.0, device=self.device)
                else:
                    logging.warning("Could not retrieve A_theta from model for spectral loss calculation.")
                
                loss = L_pred + self.spectral_loss_weight * L_spec
                train_loss_accum += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            scheduler.step()
            
        avg_loss = train_loss_accum / num_batches
        avg_pred_loss = train_pred_loss_accum / num_batches
        avg_spec_loss = train_spec_loss_accum / num_batches
        
        return {'loss': avg_loss, 'pred_loss': avg_pred_loss, 'spec_loss': avg_spec_loss}

    def _validate(self, model, epoch):
        """
        Runs validation loop for one epoch.
        
        Args:
            epoch (int): The current epoch number (for logging).
        
        Returns:
            float: The average validation loss.
        """
        model.eval()
        val_loss_accum = 0.0
        val_pred_loss_accum = 0.0
        val_spec_loss_accum = 0.0
        num_val_batches = len(self.val_loader)

        with torch.no_grad():
            # Wrap validation loader with tqdm
            val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation", leave=False)
            for batch in val_pbar:
                with autocast(device_type='cuda', enabled=True):
                    u = batch["u"].to(self.device)
                    y = batch["y"].to(self.device)
                    A_star = batch.get("A", None)
                    if A_star is not None:
                        A_star = A_star.to(self.device)

                    # Unpack the model output tuple
                    y_pred_seq, _ = model(u)

                    y_true_seq = y 
                    L_pred = F.mse_loss(y_pred_seq, y_true_seq)
                    val_pred_loss_accum += L_pred.item()

                    L_spec = torch.tensor(0.0, device=self.device)
                    A_theta = None
                    if hasattr(model, 'A_param') and hasattr(model.A_param, 'matrix'):
                        A_theta = model.A_param.matrix()
                    else:
                        logging.warning("Cannot retrieve A_theta via model.A_param.matrix().")
                    
                    if self.spectral_loss_weight > 0 and A_theta is not None and A_star is not None:
                        try:
                            L_spec = compute_spectral_loss(A_theta, A_star)
                            val_spec_loss_accum += L_spec.item()
                        except Exception as e:
                            logging.error(f"Error computing spectral loss during validation: {e}")
                            L_spec = torch.tensor(0.0, device=self.device)
                    else:
                        logging.warning("Could not retrieve A_theta from model for spectral loss calculation.")
                    
                    loss = L_pred + self.spectral_loss_weight * L_spec
                    val_loss_accum += loss.item()

                # Update validation pbar postfix
                val_pbar.set_postfix({'avg_val_loss': f'{val_loss_accum / (len(val_pbar)):.4f}'})

        avg_val_loss = val_loss_accum / num_val_batches
        avg_val_pred_loss = val_pred_loss_accum / num_val_batches
        avg_val_spec_loss = val_spec_loss_accum / num_val_batches

        logging.info(
            f"Epoch {epoch} Validation Summary: Avg Loss={avg_val_loss:.4f}, "
            f"Avg Pred Loss={avg_val_pred_loss:.4f}, Avg Spec Loss={avg_val_spec_loss:.4f}"
        )

        return {
            'loss': avg_val_loss,
            'pred_loss': avg_val_pred_loss,
            'spec_loss': avg_val_spec_loss
        }

    def validate(self, epoch: int) -> float:
        """
        Run validation on the validation dataset.

        Args:
            epoch (int): The current epoch number (for logging).

        Returns:
            float: The average validation loss.
        """
        return self._validate(self.model, epoch)['loss']
