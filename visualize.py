"""
Generates plots based on aggregated experiment results.
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import numpy as np
import logging
import torch
import os
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_results(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def plot_eigenvalues(true_eigs: Optional[np.ndarray], learned_eigs: np.ndarray, 
                     save_path: str, title: str = 'True vs. Learned Eigenvalues'):
    """Plots true and learned eigenvalues on the complex plane.
    
    Args:
        true_eigs (Optional[np.ndarray]): Ground truth eigenvalues. Can be None.
        learned_eigs (np.ndarray): Eigenvalues of the learned A matrix.
        save_path (str): Path to save the plot image.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot unit circle
    unit_circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='--', linewidth=1, label='Unit Circle')
    ax.add_patch(unit_circle)

    # Plot stability box (|Re|<0.5, |Im|<0.5)
    ax.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], 
            color='lightcoral', linestyle=':', linewidth=1, label='Stability Box (|Re|<0.5, |Im|<0.5)')

    # Plot eigenvalues
    if true_eigs is not None:
        plt.scatter(np.real(true_eigs), np.imag(true_eigs), 
                    marker='x', color='blue', s=100, label='True Eigenvalues', zorder=5)
    
    plt.scatter(np.real(learned_eigs), np.imag(learned_eigs), 
                marker='o', color='red', s=50, alpha=0.7, label='Learned Eigenvalues', zorder=4)

    # Set plot limits and aspect ratio
    max_lim = 1.1 # Slightly larger than unit circle
    if true_eigs is not None:
        max_val = np.max(np.abs(np.concatenate([true_eigs, learned_eigs]))) if len(learned_eigs)>0 else np.max(np.abs(true_eigs))
        max_lim = max(1.1, np.ceil(max_val * 10) / 10)
    elif len(learned_eigs) > 0:
         max_val = np.max(np.abs(learned_eigs))
         max_lim = max(1.1, np.ceil(max_val * 10) / 10)
        
    plt.xlim([-max_lim, max_lim])
    plt.ylim([-max_lim, max_lim])
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory

def plot_trajectory(true_y: torch.Tensor, pred_y: torch.Tensor, 
                      save_path: str, title: str = 'Sample Trajectory Comparison', 
                      max_timesteps: Optional[int] = None, output_dim_to_plot: int = 0):
    """Plots a comparison of true and predicted output trajectories.

    Args:
        true_y (torch.Tensor): Ground truth output tensor (SeqLen, OutputDim).
        pred_y (torch.Tensor): Predicted output tensor (SeqLen, OutputDim).
        save_path (str): Path to save the plot image.
        title (str): Title for the plot.
        max_timesteps (Optional[int]): Maximum number of timesteps to plot. Plots all if None.
        output_dim_to_plot (int): Index of the output dimension to plot.
    """
    if true_y.ndim > 1: true_y = true_y[:, output_dim_to_plot]
    if pred_y.ndim > 1: pred_y = pred_y[:, output_dim_to_plot]
        
    true_y_np = true_y.detach().cpu().numpy()
    pred_y_np = pred_y.detach().cpu().numpy()
    
    timesteps = len(true_y_np)
    if max_timesteps is not None:
        timesteps = min(timesteps, max_timesteps)
        true_y_np = true_y_np[:timesteps]
        pred_y_np = pred_y_np[:timesteps]
        
    time_axis = np.arange(timesteps)

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, true_y_np, label='True Output', color='blue', linestyle='-')
    plt.plot(time_axis, pred_y_np, label='Predicted Output', color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel(f'Output Dimension {output_dim_to_plot}')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory

def plot_accuracy_heatmap(results, config, output_dir):
    logging.info("Plotting Accuracy vs Mu Heatmap (Fig 3) - Placeholder")
    # Assumes experiments were run varying 'mu' (spectral mixing)
    # Requires results structured by 'mu' value.
    # Requires modification to run_experiments.py (ablations section) to handle this.
    pass

def plot_state_dim_elbow(results, config, output_dir):
    logging.info("Plotting State Dimension Elbow (Fig 4) - Placeholder")
    # Assumes experiments were run varying 'n' (state dimension)
    # Requires results structured by 'n' value.
    # Requires modification to run_experiments.py (ablations section) to handle this.
    pass

def plot_comparative_metrics(results_df, config, output_dir):
    logging.info("Plotting Comparative Metrics Bar Chart")
    try:
        metrics_to_plot = config.get('evaluation', {}).get('metrics_to_plot', 
                                ['wasserstein_distance', 'id_error_norm', 'forecast_mse_100'])
        
        df_plot = results_df[results_df['metric'].isin(metrics_to_plot)].copy()
        
        if df_plot.empty:
            logging.warning(f"No data found for metrics: {metrics_to_plot}")
            return

        plt.figure(figsize=(12, 6 * len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(len(metrics_to_plot), 1, i+1)
            metric_df = df_plot[df_plot['metric'] == metric]
            if not metric_df.empty:
                sns.barplot(x='model', y='mean', data=metric_df, capsize=.1)
                # Add error bars (std dev)
                plt.errorbar(x=metric_df['model'], y=metric_df['mean'], 
                             yerr=metric_df['std'], fmt='none', c='black', capsize=5)
                plt.title(f'Comparison: {metric}')
                plt.ylabel('Mean Value (across seeds)')
                plt.xlabel('Model Type')
                plt.xticks(rotation=45, ha='right')
            else:
                plt.title(f'Comparison: {metric} (No Data)')
        
        plt.tight_layout()
        plot_path = output_dir / "comparative_metrics.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved comparative metrics plot to {plot_path}")

    except Exception as e:
        logging.error(f"Failed to generate comparative metrics plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Plots for LTI-SSM Experiments")
    parser.add_argument('--results_path', type=str, required=True, help='Path to the aggregated JSON results file (e.g., comparative_summary_raw.json).')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the original configuration file.')
    parser.add_argument('--output_dir', type=str, default='experiment_results/plots', help='Directory to save plots.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config_path)
    results = load_results(args.results_path)
    results_df = pd.read_json(json.dumps(results), orient='index').reset_index().rename(columns={'index': 'model'}) # Needs restructuring based on actual JSON output of aggregate_results
    # Hacky way to get DF from nested dict - likely needs refinement based on actual aggregate_results output JSON structure
    # A better approach is to use the CSV output from run_experiments.py
    try:
        results_df_path = Path(args.results_path).parent / "comparative_summary.csv"
        if results_df_path.exists():
             results_df = pd.read_csv(results_df_path)
        else:
             logging.warning("Could not find comparative_summary.csv, plotting might be limited.")
             # Fallback or error
    except Exception as e:
        logging.error(f"Error loading summary CSV: {e}")

    # --- Generate Plots ---
    true_eigs = np.array([1+2j, 2+3j, 3+4j])  # Replace with actual true eigenvalues
    learned_eigs = np.array([1.1+2.1j, 2.1+3.1j, 3.1+4.1j])  # Replace with actual learned eigenvalues
    plot_eigenvalues(true_eigs, learned_eigs, output_dir / "eigenvalues.png")

    true_y = torch.randn(100, 1)  # Replace with actual true output
    pred_y = torch.randn(100, 1)  # Replace with actual predicted output
    plot_trajectory(true_y, pred_y, output_dir / "trajectory.png")

    plot_accuracy_heatmap(results, config, output_dir) # Needs refinement
    plot_state_dim_elbow(results, config, output_dir) # Needs refinement
    if 'metric' in results_df.columns: # Check if loaded from CSV
        plot_comparative_metrics(results_df, config, output_dir)
    else:
        logging.warning("Could not plot comparative metrics due to missing data format.")

    logging.info(f"Visualization script finished. Plots saved in {output_dir}")
