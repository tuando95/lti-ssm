#!/usr/bin/env python3
"""
main.py

Entry point for spectral-controlled state-space model (SSM) training and evaluation.
Loads configuration, sets up reproducibility, constructs data loaders, model, trainer,
and evaluator, then runs training and final evaluation.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
import copy
import datetime

import yaml
import torch

from utils import set_seed
from dataset_loader import DatasetLoader
# Import all potential model classes
from model import Model, get_model_A 
from baselines import RandomSSM, SimpleTransformer # Import baselines
from trainer import Trainer
from evaluation import Evaluation

# Map model type strings to classes
MODEL_REGISTRY = {
    'LTI_SSM': Model, # Assuming 'Model' is your main LTI-SSM
    'RandomSSM': RandomSSM,
    'SimpleTransformer': SimpleTransformer,
    # Add other models here if needed
}


def parse_args():
    """
    Parse command-line arguments required by run_experiments.py.
    """
    parser = argparse.ArgumentParser(
        description="Run a single LTI-SSM experiment instance."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the main YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for this specific run.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Type of model to train/evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save logs, models, etc. for this specific run.",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        required=True,
        help="Path to save the final evaluation metrics (JSON).",
    )
    parser.add_argument(
        "--param_type",
        type=str,
        default=None,
        help="Override model parameterization type for single run.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Override mu (spectral loss weight) for single run.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate for single run.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Subdirectory name for single run output.",
    )
    parser.add_argument(
        "--override",
        nargs='*',
        default=[],
        help="Override configuration parameters. Use format 'key=value'. E.g., --override training.learning_rate=0.005 model.state_dim=8",
    )
    # Add optional override arguments if needed later
    # parser.add_argument('--override', nargs='+', help='Override config params (key=value)')
    return parser.parse_args()


def setup_logging(log_dir: Path):
    """
    Configure logging to file and console for a specific run.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'run.log'
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] {%(levelname)s} %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file '{config_path}': {e}")
    if not isinstance(config, dict):
        raise ValueError(f"Invalid or empty config file: '{config_path}'")
    # Basic validation (can be extended)
    # required_sections = ["training", "data", "model"]
    # for section in required_sections:
    #     if section not in config:
    #         raise KeyError(f"Missing required section '{section}' in config.")
    return config

# Utility function to merge override dictionary into config dictionary
def merge_configs(base_config, overrides):
    """Recursively merge overrides into base_config."""
    config = copy.deepcopy(base_config)
    for key, value in overrides.items():
        if isinstance(value, dict):
            # get node or create one
            node = config.setdefault(key, {})
            # Ensure the result of the recursive call is assigned back
            config[key] = merge_configs(node, value) 
        else:
            config[key] = value
    return config


# --- Helper to select the right model instance after training --- 
def select_model_for_eval(trained_output, model_type, config, device):
    """
    Selects the appropriate model object for evaluation based on training output
    and model type.
    
    Args:
        trained_output: Output from trainer.train() (usually a dict of models or a single model)
        model_type: String identifier of the model used.
        config: Experiment configuration dict.
        device: Target device.

    Returns:
        The model object ready for evaluation.
    """
    if model_type in ['RandomSSM', 'SimpleTransformer']: # Baselines that don't need training
        logging.info(f"Instantiating pre-defined baseline: {model_type}")
        model_class = MODEL_REGISTRY[model_type]
        model_instance = model_class(config).to(device)
        # Apply DataParallel if needed (though less common for simple baselines)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            model_instance = torch.nn.DataParallel(model_instance)
        return model_instance
        
    elif isinstance(trained_output, dict): # Output from per-system training
        if not trained_output:
            raise RuntimeError("Trainer returned no models after per-system training.")
        # For evaluation, typically we want a single model representation.
        # Option 1: Evaluate each system and average metrics (complex, done by Evaluation class?)
        # Option 2: Pick one representative model (as done previously)
        # Option 3: Aggregate model parameters (if possible/meaningful)
        first_system_id = next(iter(trained_output.keys()))
        logging.warning(f"Multiple models trained (per-system). Selecting model for system '{first_system_id}' for evaluation.")
        model_instance = trained_output[first_system_id]
        # Ensure it's on the right device (might be returned from trainer on device)
        model_instance = model_instance.to(device)
        return model_instance
        
    elif isinstance(trained_output, torch.nn.Module): # Single model returned
        logging.info("Using the single model returned by the trainer.")
        model_instance = trained_output.to(device) # Ensure device
        return model_instance
        
    else:
        raise TypeError(f"Unexpected output type from trainer: {type(trained_output)}")

def run_experiment(
    config: dict,
    seed: int,
    model_type_arg: str,
    output_dir: Path,
    metrics_file: Path,
    overrides: dict = None
):
    output_dir = Path(output_dir)

    # --- Apply Overrides --- #
    if overrides:
        logging.info(f"Applying overrides within run_experiment: {overrides}")
        # --- Add Log BEFORE Merge --- #
        logging.debug(f"Config data section BEFORE merge: {config.get('data', 'MISSING!')}")
        config = merge_configs(config, overrides)
        logging.debug(f"Config data section AFTER merge: {config.get('data', 'MISSING or OVERWRITTEN!')}")
        logging.debug(f"Config after applying overrides:\n{yaml.dump(config)}")
    else:
        logging.info("No overrides provided.")

    # Setup logging for this specific run (using the final output_dir)
    setup_logging(output_dir)
    
    logging.info(f"--- Starting Experiment Run --- ")
    logging.info(f"Seed: {seed}")
    logging.info(f"Output Dir: {output_dir}")

    # Apply run-specific settings
    config['seed'] = seed
    # Override the trainer's output dir to keep run-specific models separate
    config['training'] = config.get('training', {})
    config['training']['output_dir'] = str(output_dir) # Trainer uses this
    
    logging.info(f"Using random seed = {config['seed']}")

    # Reproducibility
    set_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device) # Store device in config for others
    logging.info(f"Computation device: {device}")

    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------
    logging.info("Initializing dataset loader...")
    # Assuming DatasetLoader uses config['data'] section
    data_loader = DatasetLoader(config) # Seed is handled internally via config
    data_loader.generate_systems() # Generate all systems
    # Log dataset split sizes
    logging.info(f"Dataset generated - Train systems: {len(data_loader.train_systems)}, "
                 f"Val systems: {len(data_loader.val_systems)}, "
                 f"Test systems: {len(data_loader.test_systems)}")
    train_loader = data_loader.get_dataloader("train")
    val_loader = data_loader.get_dataloader("val")
    test_loader = data_loader.get_dataloader("test")
    logging.info(
        f"Data loaders ready - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)} batches"
    )

    # -------------------------------------------------------------------------
    # Model Selection & Training
    # -------------------------------------------------------------------------
    # Use the model_type passed as an argument
    model_type = model_type_arg.lower() # CORRECT: Use the argument passed to the function
    logging.info(f"Initializing model of type: {model_type}")

    # Get necessary dimensions from data config (used by all models)
    data_cfg = config.get('data', {})
    # Infer dimensions from a sample batch if not explicitly defined? Risky.
    # Let's assume they are defined or defaults are okay.
    # TODO: Get dimensions more robustly, maybe from dataset_loader after generation?
    # For now, assume they might be in model config or data config.
    m_dim = data_cfg.get('input_dim', config.get('model',{}).get('m_dim', 1)) # Input dim
    p_dim = data_cfg.get('output_dim', config.get('model',{}).get('p_dim', 1)) # Output dim
    # Use state_dim from model section primarily, provide a default
    n_dim = config.get('model', {}).get('state_dim', 16) # State dim (only for SSMs)

    # Add debug log before the check
    logging.debug(f"DEBUG: Checking model type. Value = '{model_type}' (type: {type(model_type)}) ")
    # --- Add Log for Data Section --- #
    logging.debug(f"Config data section before Model init: {config.get('data', 'MISSING!')}")
    model = None
    if model_type == 'lti_ssm':
        model = Model(config).to(device)
    elif model_type == 'randomssm':
        logging.info(f"Instantiating baseline model: {model_type}. It will not be trained.")
        # Use dimensions already extracted above
        logging.debug(f"RandomSSM - Using n_dim={n_dim}, m_dim={m_dim}, p_dim={p_dim}")
        model = RandomSSM(n_dim=n_dim, m_dim=m_dim, p_dim=p_dim).to(device)
    elif model_type == 'simpletransformer':
        # Extract Transformer-specific hparams using dictionary access
        logging.debug(f"Instantiating baseline model: {model_type}")
        transformer_cfg = config.get('baselines', {}).get('transformer_config', {})
        if not transformer_cfg:
            logging.error("Transformer config ('baselines.transformer_config') not found in config file!")
            raise ValueError("Missing transformer configuration")
        
        # Use dimensions already extracted above (p_dim, m_dim)
        d_model = transformer_cfg.get('d_model', 128)
        nhead = transformer_cfg.get('nhead', 8)
        num_encoder_layers = transformer_cfg.get('num_encoder_layers', 6)
        dim_feedforward = transformer_cfg.get('dim_feedforward', 2048) # Provide default
        dropout = transformer_cfg.get('dropout', 0.1)
        max_len = transformer_cfg.get('max_len', 5000)
        
        logging.debug(f"SimpleTransformer - Using p_dim={p_dim}, m_dim={m_dim}, d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}")
        
        model = SimpleTransformer(p_dim=p_dim, 
                                  d_model=d_model, 
                                  nhead=nhead, 
                                  num_encoder_layers=num_encoder_layers, 
                                  dim_feedforward=dim_feedforward, 
                                  dropout=dropout, 
                                  max_len=max_len,
                                  m_dim=m_dim, # Pass input dim as m_dim
                                  ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    logging.info(f"Model initialized: {model.__class__.__name__}")
    # Log model summary (using torchinfo or similar)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model Architecture: {model.__class__.__name__}")
    logging.info(f"Model - Total Parameters: {total_params:,}")

    # --- Training --- #
    trained_model = model # Default to the initialized model
    best_val_loss = float('inf') # Default if no training occurs

    # Use lowercase 'randomssm' to match the instantiation check
    if model_type != 'randomssm':
        logging.info(f"Preparing training for model: {model_type}")
        logging.info("Instantiating Trainer...")
        # Pass output_dir to Trainer for saving loss logs etc.

        # --- Force Disable AMP for Debugging Singularity --- #
        config['training'] = config.get('training', {}) # Ensure training section exists
        config['training']['use_amp'] = False
        logging.warning("Forcing Automatic Mixed Precision (AMP) to be DISABLED for debugging.")
        # ---------------------------------------------------- #

        trainer = Trainer(model, train_loader, val_loader, config, output_dir)

        logging.info("Starting model training...")
        # Trainer now returns the model AND the best validation loss
        # Update trained_model and best_val_loss only if training occurred
        trained_model, best_val_loss = trainer.train()
        logging.info(f"Training finished. Best validation loss achieved: {best_val_loss:.4f}")
    else:
        # This branch should now correctly execute for 'randomssm'
        logging.info(f"Skipping training for baseline model: {model_type}.")


    # --- Evaluation --- #
    logging.info("Instantiating Evaluator...")
    # Pass the potentially trained model (or initial model for RandomSSM)
    evaluator = Evaluation(trained_model, test_loader, config, str(output_dir))

    logging.info("Starting final evaluation on test set...")
    metrics = evaluator.evaluate()

    # --- Add training validation loss to metrics --- #
    if metrics is None:
        metrics = {} # Initialize if evaluation failed
    # Only add best_val_loss if training actually happened
    if model_type != 'randomssm':
        metrics['best_val_loss'] = best_val_loss
    logging.info(f"Final Evaluation Metrics: {metrics}")


    logging.info(f"--- Finished Experiment Run (Seed: {seed}, Model: {model_type}) --- ")
    # Save results
    logging.info(f"Saving final metrics to: {metrics_file}")
    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save metrics to {metrics_file}: {e}")

    logging.info("Experiment run finished.")
    return metrics # Return the computed metrics dictionary


# --- Entry Point --- 
if __name__ == "__main__":
    # Use the dedicated argument parsing function
    args = parse_args()

    # Load base configuration
    config = load_config(args.config_path)
    if not config:
        sys.exit(1) # Exit if config loading failed

    # --- Apply Overrides (Important for HPO/Ablations) ---
    if args.override:
        logging.info(f"Applying command-line overrides: {args.override}")
        # Simple override mechanism: assumes override is 'key=value'
        # More robust parsing might be needed for nested keys
        overrides = {}
        for item in args.override:
            key, value = item.split('=', 1)
            try:
                # Attempt to evaluate value (e.g., for numbers, booleans)
                # Be cautious with eval on arbitrary input
                evaluated_value = eval(value)
            except:
                evaluated_value = value # Keep as string if eval fails

            # Navigate nested keys if necessary (e.g., 'training.learning_rate')
            keys = key.split('.')
            cfg_ptr = overrides
            for k in keys[:-1]:
                cfg_ptr = cfg_ptr.setdefault(k, {})
            cfg_ptr[keys[-1]] = evaluated_value
            logging.info(f"  Override: {key} = {evaluated_value}")
    else:
        overrides = None

    # --- Set up output directory --- #
    # Use the output_dir passed from run_experiments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging for this specific run
    setup_logging(output_dir)

    # Add relevant args back into config for easy access within run_experiment
    config['seed'] = args.seed
    config['output_dir'] = str(output_dir) # Pass output dir via config
    config['model_type'] = args.model_type # Pass model type via config
    config['config_path'] = args.config_path # Store original path

    # --- Run the Experiment --- #
    logging.info(f"Starting experiment for seed {args.seed}, model {args.model_type}...")
    logging.info(f"Full configuration:\n{yaml.dump(config, indent=2)}")

    try:
        # Pass the combined config and the specified seed
        run_experiment(
            config=config, 
            seed=args.seed, 
            model_type_arg=args.model_type, # Pass parsed model type
            output_dir=output_dir,
            metrics_file=Path(args.metrics_file), # Pass parsed metrics file path
            overrides=overrides # Pass parsed overrides
            )
        logging.info(f"Experiment run with seed {args.seed} completed successfully.")
    except Exception as e:
        logging.error(f"Experiment run with seed {args.seed} failed: {e}", exc_info=True)

    logging.info("All experiment runs finished.")
