"""
Main script to orchestrate experiments based on config.yaml.
Handles multiple seeds, model types, evaluation, aggregation, and visualization triggering.
"""

import yaml
import subprocess
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import logging
import argparse
from pathlib import Path
import copy
import functools # For partial function application in HPO
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns # Added for plotting
import itertools # For combinations
try:
    import optuna
except ImportError:
    logging.warning("Optuna not found. HPO functionality will be disabled. Run 'pip install optuna' to enable it.")
    optuna = None # Set optuna to None if not installed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_single_experiment(seed, model_type, config, output_dir):
    """Runs a single train/eval experiment for a given seed and model type."""
    logging.info(f"Running experiment: seed={seed}, model={model_type}")
    run_output_dir = output_dir / f"seed_{seed}_model_{model_type}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_output_dir / "metrics.json"

    # Prepare command line arguments for main.py
    cmd = [
        "python", "main.py",
        "--config_path", str(config['config_path']),
        "--seed", str(seed),
        "--model_type", model_type,
        "--output_dir", str(run_output_dir),
        "--metrics_file", str(metrics_file)
    ]
    
    # --- Add Overrides --- #
    # Check if the passed config differs from a base config or contains overrides
    # This is a simplified approach: assuming overrides are directly in the top-level config dict for now.
    # A more robust way would involve comparing against the original base_config.
    # Example overrides (adjust keys based on your actual config structure):
    possible_overrides = {
        'training.learning_rate': config.get('training', {}).get('learning_rate'),
        'training.weight_decay': config.get('training', {}).get('weight_decay'),
        'training.spectral_loss_weight': config.get('training', {}).get('spectral_loss_weight'),
        'model.state_dim': config.get('model', {}).get('state_dim'),
        'model.parameterization_type': config.get('model', {}).get('parameterization_type'),
        'model.low_rank_r': config.get('model', {}).get('low_rank_r'),
        'data.noise_sigma': config.get('data', {}).get('noise_sigma'),
        # Add other potential overrides from HPO/ablations here
    }

    override_args = []
    for key, value in possible_overrides.items():
        # Check if value exists AND is NOT a list/tuple (expected override format is key=single_value)
        if value is not None and not isinstance(value, (list, tuple)):
            # Format value correctly for command line (e.g., handle bools)
            if isinstance(value, bool):
                str_value = str(value).lower() # True -> true
            else:
                str_value = str(value)
            override_args.extend(["--override", f"{key}={str_value}"])
        elif isinstance(value, (list, tuple)):
             # Log a warning that a list was found where a single value was expected for override
             logging.warning(f"Skipping override for '{key}': Expected single value but found list/tuple: {value}. Check HPO setup or config overrides.")

    if override_args:
        cmd.extend(override_args)
        logging.debug(f"Added overrides to command: {' '.join(override_args)}")

    # Log the final command
    logging.debug(f"Executing command: {' '.join(cmd)}")

    try:
        # Modify subprocess.run to allow stderr through for tqdm, capture stdout
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=None, text=True, cwd=Path(__file__).parent)
        logging.info(f"Run seed={seed}, model={model_type} completed successfully.")
        # Log stdout (captured)
        logging.debug(f"STDOUT:\n{process.stdout}")

        # Check if metrics file was created
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # --- Add study group information --- #
            try:
                # Infer study group from the parent directory name (e.g., 'comparative', 'ablation_X')
                study_group = run_output_dir.parent.parent.name 
                metrics_data['study_group'] = study_group
                logging.debug(f"Added study_group='{study_group}' to metrics for {model_type} seed {seed}")
            except Exception as e:
                logging.warning(f"Could not determine study_group for {run_output_dir}: {e}. Setting to 'unknown'.")
                metrics_data['study_group'] = 'unknown'
            # ----------------------------------- #
                
            return metrics_data # Return the dictionary containing metrics
        else:
            logging.error(f"Metrics file not found for seed={seed}, model={model_type} at {metrics_file}")
            return None

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running seed={seed}, model={model_type}: {e}")
        # Log stdout/stderr if available in the exception
        if e.stdout:
            logging.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
             logging.error(f"STDERR:\n{e.stderr}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during run seed={seed}, model={model_type}: {e}")
        return None

def aggregate_results(all_run_metrics, output_dir):
    """Aggregates metrics from all runs into a pandas DataFrame and saves it."""
    if not all_run_metrics:
        logging.warning("No metrics data to aggregate.")
        return None

    try:
        # Convert list of dicts to DataFrame
        results_df = pd.DataFrame(all_run_metrics)

        # Optional: Add derived columns or perform initial cleaning if needed
        # e.g., results_df['error_ratio'] = results_df['error'] / results_df['baseline']

        # Save the aggregated DataFrame
        output_csv_path = Path(output_dir) / "aggregated_results.csv"
        results_df.to_csv(output_csv_path, index=False)
        logging.info(f"Aggregated results saved to '{output_csv_path}'")

        # --- Perform Statistical Tests (Paired T-tests) --- # 
        # Compare models pairwise on key metrics for comparative runs
        comparative_df = results_df[results_df['study_group'] == 'comparative'].copy()
        if not comparative_df.empty:
            key_metrics_for_test = [
                'final_test_loss', # Example primary loss
                'id_error_norm', # Example spectral metric
                'wasserstein_distance', # Alternative spectral metric
                'forecast_mse_100' # Example forecast metric
            ]
            for metric in key_metrics_for_test:
                if metric in comparative_df.columns:
                     _perform_paired_t_test(comparative_df, 'model_type', metric, 'seed', alpha=0.05)
                else:
                     logging.warning(f"Metric '{metric}' not found in results, skipping t-test.")
        else:
            logging.warning("No comparative study results found, skipping paired t-tests.")

        # --- Calculate Variance for Initialization Ablation --- #
        ablation_df = results_df[results_df['study_group'] == 'ablation'].copy()
        if not ablation_df.empty:
            # Identify initialization ablation parameter (heuristic based on name)
            init_param = None
            potential_init_params = [p for p in ablation_df['ablation_parameter'].unique() if 'init' in str(p).lower()]
            if len(potential_init_params) == 1:
                init_param = potential_init_params[0]
                logging.info(f"Identified initialization ablation parameter: '{init_param}'")
            elif len(potential_init_params) > 1:
                 logging.warning(f"Multiple potential initialization ablation parameters found: {potential_init_params}. Cannot reliably calculate variance. Please refine identification logic.")
            else:
                logging.info("No parameter name containing 'init' found in ablation studies. Skipping initialization variance calculation.")
                
            if init_param:
                init_ablation_df = ablation_df[ablation_df['ablation_parameter'] == init_param]
                # Metrics to calculate variance for
                variance_metrics = ['final_test_loss', 'wasserstein_distance', 'hungarian_spectral_loss'] # Adjust as needed
                
                grouped = init_ablation_df.groupby(['model_type', 'ablation_value'])
                
                logging.info(f"--- Variance Report for Initialization Ablation ('{init_param}') --- ")
                for (model_type, init_value), group in grouped:
                    logging.info(f"  Model: {model_type}, Init Setting: {init_value}")
                    for metric in variance_metrics:
                        if metric in group.columns:
                            variance = group[metric].var()
                            mean_val = group[metric].mean()
                            std_val = group[metric].std()
                            count = group[metric].count()
                            logging.info(f"    Metric: {metric:<25} | Var: {variance:.4e} | Mean: {mean_val:.4f} | Std: {std_val:.4f} (n={count})")
                        #else:
                        #    logging.debug(f"    Metric: {metric} not found for this group.")
                logging.info("--- End Variance Report --- ")

        # --- Generate Summary Table --- #
        try:
            _generate_summary_table(results_df, Path(output_dir))
        except Exception as e:
            logging.error(f"Failed to generate summary table: {e}", exc_info=True)
        
        return results_df

    except Exception as e:
        logging.error(f"Error aggregating results: {e}", exc_info=True)
        return None

def _generate_summary_table(results_df: pd.DataFrame, output_dir: Path):
    """Generates a formatted summary table (like Table 1) from aggregated results."""
    logging.info("Generating summary table (Table 1)...")
    comparative_df = results_df[results_df['study_group'] == 'comparative'].copy()
    
    if comparative_df.empty:
        logging.warning("No comparative results found, cannot generate summary table.")
        return

    # Define key metrics to include in the table
    metrics_to_summarize = [
        'final_test_loss',
        'wasserstein_distance',
        'id_error_norm',
        'hungarian_spectral_loss',
        'forecast_mse_1', 
        'forecast_mse_10', 
        'forecast_mse_100'
    ]
    
    # Filter out metrics not present in the dataframe
    available_metrics = [m for m in metrics_to_summarize if m in comparative_df.columns]
    if not available_metrics:
        logging.warning("None of the specified key metrics found in results. Skipping summary table.")
        return
        
    # Calculate mean and std, grouping by model type
    summary = comparative_df.groupby('model_type')[available_metrics].agg(['mean', 'std'])

    # Format as 'mean ± std'
    summary_formatted = pd.DataFrame(index=summary.index)
    for metric in available_metrics:
        mean_col = (metric, 'mean')
        std_col = (metric, 'std')
        # Handle cases where std might be NaN (e.g., only one seed run)
        summary_formatted[metric] = summary.apply(
            lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}" if pd.notna(row[std_col]) else f"{row[mean_col]:.3f}",
            axis=1
        )
        
    # Rename columns for clarity (optional)
    summary_formatted.columns = [col.replace('_', ' ').title() for col in summary_formatted.columns]
    
    # Save as Markdown
    table_path = output_dir / "summary_table.md"
    try:
        summary_formatted.to_markdown(table_path)
        logging.info(f"Summary table saved to {table_path.name}")
    except Exception as e:
        logging.error(f"Failed to save summary table to markdown: {e}")

def visualize_results(results_df, viz_config, eval_config, output_dir):
    """Creates and saves plots based on the aggregated results DataFrame."""
    if results_df is None or results_df.empty:
        logging.warning("No results DataFrame to visualize. Skipping visualization.")
        return
    if not viz_config:
        logging.warning("No visualization configuration found. Skipping visualization.")
        return

    # Ensure plotting libraries are available
    if plt is None or sns is None:
        logging.error("Matplotlib or Seaborn not found. Cannot create visualizations. Install them: pip install matplotlib seaborn")
        return

    metrics_to_plot = viz_config.get('metrics_to_plot', [])
    if not metrics_to_plot:
        logging.info("No 'metrics_to_plot' specified in visualization config. Skipping plotting.")
        return

    plot_output_dir = Path(output_dir) / "plots"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving plots to: {plot_output_dir}")

    # Determine grouping factors (e.g., model_type, ablation study/value)
    grouping_factors = ['model_type']
    if 'ablation_study' in results_df.columns:
        grouping_factors.append('ablation_study')
        # Could also group by ablation_value, depending on desired plot type

    # Example: Create boxplots for each specified metric, grouped by model_type
    for metric in metrics_to_plot:
        if metric not in results_df.columns:
            logging.warning(f"Metric '{metric}' specified for plotting not found in results. Skipping.")
            continue

        plt.figure(figsize=(12, 7))
        # Basic boxplot grouped by model type
        sns.boxplot(data=results_df, x='model_type', y=metric)
        plt.title(f'{metric} by Model Type (Across Seeds)')
        plt.xlabel('Model Type')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = plot_output_dir / f"boxplot_{metric}_by_model.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot: {plot_path.name}")

        # If ablation results exist, create plots faceted by ablation study
        if 'ablation_study' in results_df.columns:
            # Ensure the ablation parameter/value columns exist
            if 'ablation_parameter' in results_df.columns and 'ablation_value' in results_df.columns:
                 # Use seaborn's FacetGrid or catplot for more complex visualizations
                 try:
                     # Try creating a catplot (handles categorical x-axis well)
                     g = sns.catplot(
                         data=results_df,
                         x='ablation_value', 
                         y=metric,
                         col='ablation_study', # Create subplots for each study
                         hue='model_type',     # Color by model type within each plot
                         kind='box',           # Use boxplot type
                         sharex=False,         # Allow different x-axes per study
                         col_wrap=2,           # Wrap facets into 2 columns
                         height=4, aspect=1.5
                     )
                     g.fig.suptitle(f'{metric} Ablation Study Results', y=1.03) # Add overall title
                     g.set_titles(col_template="{col_name}")
                     g.set_axis_labels("Ablation Value", metric)
                     # Adjust x-tick labels if they are numeric but treated as categorical
                     for ax in g.axes.flat:
                         try:
                             # Attempt to convert labels to numeric for sorting/nicer display if possible
                             labels = [item.get_text() for item in ax.get_xticklabels()]
                             numeric_labels = pd.to_numeric(labels, errors='ignore')
                             if pd.api.types.is_numeric_dtype(numeric_labels):
                                 ax.set_xticks(range(len(numeric_labels)))
                                 ax.set_xticklabels([f"{float(l):.1e}" if abs(float(l)) < 1e-2 or abs(float(l)) > 1e3 else str(float(l)) for l in numeric_labels])
                             ax.tick_params(axis='x', rotation=45)
                         except Exception: # Handle cases where conversion fails gracefully
                             ax.tick_params(axis='x', rotation=45)
                     
                     plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for title
                     plot_path = plot_output_dir / f"catplot_{metric}_ablations.png"
                     plt.savefig(plot_path)
                     plt.close(g.fig) # Close the figure associated with FacetGrid
                     logging.info(f"Saved plot: {plot_path.name}")
                 except Exception as e:
                     logging.error(f"Failed to create catplot for metric {metric} during ablation viz: {e}", exc_info=True)
                     # Fallback or alternative plot maybe?
            else:
                logging.warning(f"Ablation parameter/value columns not found for detailed ablation plot of metric {metric}.")

    # --- Specific Ablation Plots (Fig 3 & 4 from paper outline) ---
    if 'ablation_study' in results_df.columns:
        # Fig 3: Heatmap of Accuracy vs. Mu (spectral_loss_weight)
        mu_ablation_param = 'training.spectral_loss_weight'
        # Assuming an accuracy metric exists, e.g., 'final_test_accuracy'. 
        # If not, this needs adjustment based on available metrics.
        accuracy_metric = 'final_test_accuracy' # Placeholder - VERIFY METRIC NAME
        mu_df = results_df[
            (results_df['ablation_parameter'] == mu_ablation_param) &
            (results_df['study_group'] == 'ablation')
        ]
        if not mu_df.empty and accuracy_metric in mu_df.columns:
            logging.info(f"Generating Heatmap: {accuracy_metric} vs. {mu_ablation_param}")
            try:
                # Pivot table: index=model_type, columns=mu_value, values=accuracy (mean over seeds)
                pivot_df = mu_df.pivot_table(
                    index='model_type',
                    columns='ablation_value', 
                    values=accuracy_metric,
                    aggfunc='mean' 
                )
                plt.figure(figsize=(10, 7))
                sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis")
                plt.title(f'{accuracy_metric} vs. Spectral Loss Weight ($\mu$)')
                plt.xlabel('Spectral Loss Weight ($\mu$)')
                plt.ylabel('Model Type')
                plt.tight_layout()
                plot_path = plot_output_dir / f"heatmap_{accuracy_metric}_vs_mu.png"
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Saved plot: {plot_path.name}")
            except Exception as e:
                logging.error(f"Failed to create heatmap for {accuracy_metric} vs {mu_ablation_param}: {e}", exc_info=True)
        elif mu_ablation_param in results_df['ablation_parameter'].unique():
             logging.warning(f"Found ablation data for {mu_ablation_param}, but metric '{accuracy_metric}' not found. Skipping heatmap.")

        # Fig 4: Elbow Plot: State Dimension vs. L_spec (or related metric)
        state_dim_ablation_param = 'model.state_dim'
        # Use a spectral-related metric like id_error_norm or wasserstein_distance
        spectral_metric_for_elbow = 'id_error_norm' # Placeholder - VERIFY METRIC NAME
        state_dim_df = results_df[
            (results_df['ablation_parameter'] == state_dim_ablation_param) &
            (results_df['study_group'] == 'ablation')
        ]
        if not state_dim_df.empty and spectral_metric_for_elbow in state_dim_df.columns:
            logging.info(f"Generating Elbow Plot: {spectral_metric_for_elbow} vs. {state_dim_ablation_param}")
            try:
                # Ensure state dim values are numeric for plotting
                state_dim_df['ablation_value_numeric'] = pd.to_numeric(state_dim_df['ablation_value'])
                
                plt.figure(figsize=(10, 7))
                sns.lineplot(
                    data=state_dim_df, 
                    x='ablation_value_numeric', 
                    y=spectral_metric_for_elbow, 
                    hue='model_type', 
                    marker='o', 
                    errorbar='sd' # Show standard deviation across seeds
                )
                plt.title(f'{spectral_metric_for_elbow} vs. State Dimension (n)')
                plt.xlabel('State Dimension (n)')
                plt.ylabel(spectral_metric_for_elbow)
                plt.xscale('log', base=2) # Often state dims are powers of 2
                plt.xticks(state_dim_df['ablation_value_numeric'].unique(), labels=state_dim_df['ablation_value_numeric'].unique())
                plt.grid(True, which="both", ls="--", alpha=0.6)
                plt.legend(title='Model Type')
                plt.tight_layout()
                plot_path = plot_output_dir / f"elbow_{spectral_metric_for_elbow}_vs_state_dim.png"
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Saved plot: {plot_path.name}")
            except Exception as e:
                logging.error(f"Failed to create elbow plot for {spectral_metric_for_elbow} vs {state_dim_ablation_param}: {e}", exc_info=True)
        elif state_dim_ablation_param in results_df['ablation_parameter'].unique():
             logging.warning(f"Found ablation data for {state_dim_ablation_param}, but metric '{spectral_metric_for_elbow}' not found. Skipping elbow plot.")

    # --- Fig 1: Aggregated Eigenvalue Plot --- #
    try:
        _plot_aggregated_eigenvalues(base_output_dir, plot_output_dir, results_df)
    except Exception as e:
        logging.error(f"Failed to generate aggregated eigenvalue plot (Fig 1): {e}", exc_info=True)
        
    # --- Fig 2: Aggregated Loss Trajectories Plot --- #
    try:
        _plot_aggregated_loss_trajectories(base_output_dir, plot_output_dir, results_df)
    except Exception as e:
        logging.error(f"Failed to generate aggregated loss trajectory plot (Fig 2): {e}", exc_info=True)

    logging.info("Finished creating visualizations.")

def _plot_aggregated_eigenvalues(base_output_dir: Path, plot_output_dir: Path, results_df: pd.DataFrame):
    """Helper to find eigenvalue files, aggregate, and plot Fig 1."""
    logging.info("Attempting to generate Aggregated Eigenvalue Plot (Fig 1)...")
    eigenvalue_files = list(base_output_dir.rglob('**/eigenvalues.npz'))
    
    if not eigenvalue_files:
        logging.warning("No 'eigenvalues.npz' files found. Skipping aggregated eigenvalue plot.")
        return

    all_eigs_data = {'model_type': [], 'seed': [], 'type': [], 'eigenvalue': []}
    relevant_runs = results_df[results_df['study_group'] == 'comparative'] # Focus on comparative runs for this plot usually
    if relevant_runs.empty:
        logging.warning("No 'comparative' runs found in results_df, using all runs for eigenvalue plot.")
        relevant_runs = results_df # Fallback to all runs
        
    # Map run_output_dir back to model_type and seed using results_df
    dir_to_info = {row['run_output_dir']: (row['model_type'], row['seed'])
                   for _, row in relevant_runs.iterrows() if 'run_output_dir' in row}

    for f_path in eigenvalue_files:
        run_dir_str = str(f_path.parent)
        if run_dir_str in dir_to_info:
            model_type, seed = dir_to_info[run_dir_str]
            try:
                data = np.load(f_path)
                true_eigs = data['true'].flatten() # Flatten: (samples*n,)
                pred_eigs = data['pred'].flatten() # Flatten: (samples*n,)
                
                for eig in true_eigs:
                    all_eigs_data['model_type'].append(model_type)
                    all_eigs_data['seed'].append(seed)
                    all_eigs_data['type'].append('True')
                    all_eigs_data['eigenvalue'].append(eig)
                for eig in pred_eigs:
                     all_eigs_data['model_type'].append(model_type)
                     all_eigs_data['seed'].append(seed)
                     all_eigs_data['type'].append('Predicted')
                     all_eigs_data['eigenvalue'].append(eig)
            except Exception as e:
                logging.warning(f"Could not load or process eigenvalue file {f_path}: {e}")
        else:
            logging.warning(f"Could not map eigenvalue file {f_path} back to a run in results_df.")

    if not all_eigs_data['eigenvalue']:
        logging.warning("No valid eigenvalue data loaded. Skipping plot.")
        return
        
    eigs_df = pd.DataFrame(all_eigs_data)
    # Extract real and imaginary parts for plotting
    eigs_df['real'] = eigs_df['eigenvalue'].apply(np.real)
    eigs_df['imag'] = eigs_df['eigenvalue'].apply(np.imag)

    # Create the plot (e.g., one plot per model type, overlaying true and predicted)
    model_types = eigs_df['model_type'].unique()
    num_models = len(model_types)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), squeeze=False)
    fig.suptitle('Fig 1: Aggregated Ground Truth vs. Learned Eigenvalues', fontsize=16)
    
    for i, model_type in enumerate(model_types):
        ax = axes[0, i]
        model_df = eigs_df[eigs_df['model_type'] == model_type]
        true_df = model_df[model_df['type'] == 'True']
        pred_df = model_df[model_df['type'] == 'Predicted']
        
        # Plot true eigenvalues (pooled across seeds/samples)
        ax.scatter(true_df['real'], true_df['imag'], label='True $\Lambda(A^*)$', 
                   marker='x', color='black', alpha=0.6, s=50)
        # Plot predicted eigenvalues (pooled across seeds/samples)
        ax.scatter(pred_df['real'], pred_df['imag'], label='Predicted $\Lambda(A_\theta)$', 
                   marker='o', facecolors='none', edgecolors='red', alpha=0.6, s=50)
                   
        ax.set_title(f'{model_type}')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        ax.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plot_path = plot_output_dir / "fig1_aggregated_eigenvalues.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved aggregated eigenvalue plot (Fig 1) to {plot_path.name}")

def _plot_aggregated_loss_trajectories(base_output_dir: Path, plot_output_dir: Path, results_df: pd.DataFrame):
    """Helper to find loss files, aggregate, and plot Fig 2."""
    logging.info("Attempting to generate Aggregated Loss Trajectories Plot (Fig 2)...")
    loss_files = list(base_output_dir.rglob('**/losses.csv'))

    if not loss_files:
        logging.warning("No 'losses.csv' files found. Skipping aggregated loss trajectories plot.")
        return

    all_loss_data = []
    relevant_runs = results_df[results_df['study_group'] == 'comparative'] # Focus on comparative runs
    if relevant_runs.empty:
        logging.warning("No 'comparative' runs found in results_df, using all runs for loss plot.")
        relevant_runs = results_df # Fallback

    # Map run_output_dir back to model_type and seed using results_df
    dir_to_info = {row['run_output_dir']: (row['model_type'], row['seed'])
                   for _, row in relevant_runs.iterrows() if 'run_output_dir' in row}

    for f_path in loss_files:
        run_dir_str = str(f_path.parent)
        if run_dir_str in dir_to_info:
            model_type, seed = dir_to_info[run_dir_str]
            try:
                df = pd.read_csv(f_path)
                df['model_type'] = model_type
                df['seed'] = seed
                all_loss_data.append(df)
            except Exception as e:
                logging.warning(f"Could not load or process loss file {f_path}: {e}")
        else:
            logging.warning(f"Could not map loss file {f_path} back to a run in results_df.")

    if not all_loss_data:
        logging.warning("No valid loss data loaded. Skipping plot.")
        return

    losses_df = pd.concat(all_loss_data, ignore_index=True)

    # --- Plotting --- #
    metrics_to_plot = [
        ('Train Loss', 'train_loss', 'val_loss'),
        ('Spectral Loss', 'train_spec_loss', 'val_spec_loss'),
        ('Prediction Loss', 'train_pred_loss', 'val_pred_loss'),
    ]
    num_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5), squeeze=False)
    fig.suptitle('Fig 2: Aggregated Loss Trajectories (Mean ± SD over Seeds)', fontsize=16)

    for i, (title, train_metric, val_metric) in enumerate(metrics_to_plot):
        ax = axes[0, i]
        if train_metric in losses_df.columns and val_metric in losses_df.columns:
            # Plot Train Loss
            sns.lineplot(data=losses_df, x='epoch', y=train_metric, hue='model_type', 
                         errorbar='sd', ax=ax, legend= (i == num_plots -1), linestyle='-') # Only last legend
            # Plot Validation Loss
            sns.lineplot(data=losses_df, x='epoch', y=val_metric, hue='model_type', 
                         errorbar='sd', ax=ax, legend=False, linestyle='--')
            
            ax.set_title(f'{title}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_yscale('log') # Often helpful for losses
            ax.grid(True, which="both", ls="--", alpha=0.6)
            # Custom legend entries if needed
            if i == num_plots -1:
                handles, labels = ax.get_legend_handles_labels()
                # Need to figure out how to label train/val lines clearly
                # Maybe iterate models and plot train/val explicitly? TODO: Improve legend clarity
                ax.legend(title='Model Type') # Simple legend for now
            else:
                 ax.get_legend().remove()
        else:
            ax.set_title(f'{title} (Data Missing)')
            ax.text(0.5, 0.5, 'Metric data not found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = plot_output_dir / "fig2_aggregated_loss_trajectories.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved aggregated loss trajectories plot (Fig 2) to {plot_path.name}")

def hpo_objective(trial, model_type, base_config, base_config_path, hpo_output_dir):
    """Objective function for Optuna HPO."""
    # Create a deep copy of the base config to modify for this trial
    current_config = copy.deepcopy(base_config)
    hpo_config = current_config.get('hpo', {})
    search_space = hpo_config.get('search_space', {})

    # Suggest hyperparameters based on the search space defined in config
    suggested_params = {}
    for param_key, definition in search_space.items():
        param_type = definition[0].lower()
        try:
            if param_type == 'float':
                low, high = definition[1], definition[2]
                log_scale = len(definition) > 3 and definition[3].lower() == 'log'
                value = trial.suggest_float(param_key, low, high, log=log_scale)
            elif param_type == 'int':
                low, high = definition[1], definition[2]
                value = trial.suggest_int(param_key, low, high)
            elif param_type == 'categorical':
                choices = definition[1:]
                value = trial.suggest_categorical(param_key, choices)
            else:
                logging.warning(f"Unsupported parameter type '{param_type}' in HPO search space for key '{param_key}'. Skipping.")
                continue
            suggested_params[param_key] = value
            # Update the config copy
            set_nested_value(current_config, param_key, value)
        except Exception as e:
            logging.error(f"Error suggesting parameter {param_key} with definition {definition}: {e}")
            # Return a large value to indicate failure
            return float('inf')

    # Define output dir and temp config path for this specific trial
    trial_output_dir = hpo_output_dir / f"trial_{trial.number}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = trial_output_dir / f"temp_config_trial_{trial.number}.yaml"
    save_config(current_config, temp_config_path)

    # Log trial information
    logging.info(f"--- Optuna Trial {trial.number} Start ---")
    logging.info(f"Hyperparameters: {trial.params}")
    # Optionally log the full config for the trial (can be verbose)
    # logging.debug(f"Trial {trial.number} Config:\n{yaml.dump(current_config, indent=2)}")

    # Run the experiment with the current hyperparameters
    # Ensure run_single_experiment uses the trial's output dir and config
    # Use the base seed for HPO trials for consistency, default to 0
    hpo_seed = base_config.get('seed', 0)
    logging.info(f"Using seed {hpo_seed} for HPO trial {trial.number}")
    metrics = run_single_experiment(hpo_seed, model_type, current_config, trial_output_dir)

    if metrics is None:
        logging.error(f"Trial {trial.number} failed to produce metrics. Returning high value.")
        return float('inf')  # Indicate failure to Optuna

    # Extract the metric to optimize
    metric_key = base_config['hpo']['metric']
    metric_value = metrics.get(metric_key)

    if metric_value is None:
        logging.error(f"Trial {trial.number} completed but metric '{metric_key}' not found in results. Returning high value.")
        logging.debug(f"Available metrics: {metrics.keys()}")
        return float('inf') # Indicate failure

    logging.info(f"--- Optuna Trial {trial.number} End --- Metric ({metric_key}): {metric_value:.6f} ---")

    # Optuna minimizes by default, negate if maximizing
    direction = base_config['hpo'].get('direction', 'minimize')
    if direction.lower() == 'maximize':
        return -metric_value
    else:
        return metric_value

def run_hpo(base_config, base_config_path, models_to_run, output_dir):
    """Performs Hyperparameter Optimization using Optuna for specified models."""
    if optuna is None:
        logging.error("Optuna is not installed. Cannot run HPO.")
        return {}

    hpo_config = base_config.get('hpo', {})
    if not hpo_config.get('enabled', False):
        logging.info("HPO is disabled in the configuration. Skipping.")
        return {}

    logging.info("--- Starting Hyperparameter Optimization --- ")
    num_trials = hpo_config.get('num_trials', 20) # Default to 20 trials if not specified
    hpo_output_dir = Path(output_dir) / "hpo_studies"
    hpo_output_dir.mkdir(parents=True, exist_ok=True)

    best_hps_per_model = {}

    for model_type in models_to_run:
        logging.info(f"  Running HPO for model: {model_type}")
        model_hpo_output_dir = hpo_output_dir / model_type
        model_hpo_output_dir.mkdir(parents=True, exist_ok=True)

        # Use functools.partial to pass fixed arguments to the objective function
        objective_with_args = functools.partial(
            hpo_objective,
            model_type=model_type,
            base_config=base_config,
            base_config_path=base_config_path,
            hpo_output_dir=model_hpo_output_dir
        )

        # Create and run the Optuna study
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective_with_args, n_trials=num_trials)
            best_params = study.best_trial.params
            best_value = study.best_trial.value
            logging.info(f"  Best HPs found for {model_type}: {best_params} (Validation Loss: {best_value})")
            best_hps_per_model[model_type] = best_params
        except Exception as e:
            logging.error(f"  Optuna study failed for model {model_type}: {e}")
            best_hps_per_model[model_type] = None # Indicate failure for this model

    logging.info("--- Finished Hyperparameter Optimization --- ")
    return best_hps_per_model

def set_nested_value(d, keys, value):
    keys_list = keys.split('.')
    for key in keys_list[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys_list[-1]] = value
    return True

def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f, indent=2)

def run_ablations(base_config, base_config_path, models_to_run, num_seeds, output_dir):
    """Runs ablation studies defined in the 'ablation_studies' config section."""
    logging.info("Starting ablation study runs...")
    ablation_studies = base_config.get('ablation_studies', [])
    if not ablation_studies:
        logging.warning("No 'ablation_studies' defined in config. Skipping.")
        return []

    all_ablation_metrics = []

    for study in ablation_studies:
        study_name = study.get('name', 'unnamed_ablation')
        parameter_key = study.get('parameter')
        values = study.get('values')

        if not parameter_key or not values:
            logging.warning(f"Skipping invalid ablation study definition: {study}")
            continue

        logging.info(f"--- Running Ablation Study: {study_name} (Parameter: {parameter_key}) ---")
        study_output_dir = Path(output_dir) / study_name
        study_output_dir.mkdir(parents=True, exist_ok=True)

        for value in values:
            logging.info(f"  Testing value: {value}")
            # Create a deep copy of the base config to modify
            current_config = copy.deepcopy(base_config)

            # Set the parameter value in the copied config
            success = set_nested_value(current_config, parameter_key, value)
            if not success:
                logging.error(f"    Failed to set parameter {parameter_key} to {value}. Skipping this value.")
                continue

            # Define output dir and temp config path for this specific setting
            setting_output_dir = study_output_dir / f"value_{value}"
            setting_output_dir.mkdir(parents=True, exist_ok=True)
            temp_config_path = setting_output_dir / f"temp_config_{study_name}_{value}.yaml"
            save_config(current_config, temp_config_path)

            # Determine the model type for this specific ablation run
            # Always use the primary model type defined in the config for the main script call.
            # The specific ablation parameter (like parameterization_type) will be passed via overrides.
            model_type_for_run = base_config['model']['type']

            # Run the experiment with the specific ablation config
            for seed in range(num_seeds):
                for model_type in models_to_run:
                    metrics = run_single_experiment(seed, model_type_for_run, current_config, setting_output_dir)
                    if metrics:
                        # Add identifiers for this ablation run
                        metrics['study_group'] = 'ablation'
                        metrics['ablation_study'] = study_name
                        metrics['ablation_parameter'] = parameter_key
                        metrics['ablation_value'] = value
                        all_ablation_metrics.append(metrics)
            
            # Optional: Clean up temp config file
            # temp_config_path.unlink(missing_ok=True)

        logging.info(f"--- Finished Ablation Study: {study_name} ---")

    logging.info("Finished all ablation study runs.")
    return all_ablation_metrics

# --- Main Execution Logic --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LTI-SSM Experiments")
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config_path)
    config['config_path'] = args.config_path # Store path for subordinate scripts

    exp_config = config.get('experiment', {})
    num_seeds = exp_config.get('num_seeds', 1) # Default to 1 seed if not specified
    base_output_dir = Path(exp_config.get('output_dir', 'experiment_results'))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = [config.get('model', {}).get('type', 'LTI_SSM')] # Start with the main model
    models_to_run.extend(config.get('baselines', {}).get('types', []))

    all_run_metrics = [] # Stores metrics dict from each successful run
    trained_models_info = {} # Stores paths or identifiers for trained models

    # --- HPO --- #
    # Note: We run HPO first to potentially find best HPs before other runs,
    # but currently, these HPs aren't automatically used in subsequent runs.
    # This would require modifying the config before ablations/comparative runs.
    best_hps = {}
    if optuna is not None: # Only run if optuna is installed
        hpo_config = config.get('hpo', {})
        if hpo_config.get('enabled', False):
            hpo_output_dir = base_output_dir / "hpo"
            best_hps = run_hpo(config, args.config_path, models_to_run, hpo_output_dir)
            
            # --- Apply Best HPs Found --- #
            if best_hps:
                logging.info("Applying best hyperparameters found during HPO to the main config...")
                config_updated = False
                for model_type, hps in best_hps.items():
                    if hps: # Check if HPO succeeded for this model
                        logging.info(f"  Applying HPs for {model_type}: {hps}")
                        # Need to handle model-specific config updates
                        # Assuming HPs directly map to keys in the main config or model-specific sections
                        # Example: If HPs are like {'learning_rate': 0.001, 'weight_decay': 0.0001}
                        # And these apply globally or specifically to 'model_type'
                        # We need a robust way to map these back. 
                        # For now, let's assume HPs are top-level keys in the config or within model specific blocks
                        
                        # Simplistic approach: Update top-level or model block if exists
                        for hp_key, hp_value in hps.items():
                            # Try setting globally first
                            if not set_nested_value(config, hp_key, hp_value):
                                 # Try setting within model-specific config if applicable
                                 # This requires knowing the config structure for each model type
                                 logging.warning(f"Could not directly set HPO parameter {hp_key} globally. Model-specific logic might be needed.") 
                            else:
                                config_updated = True
                                logging.info(f"    Set {hp_key} = {hp_value}")
                if config_updated:
                     logging.info("Main config updated with HPO results.")
                     # Optional: Save the updated config for reference
                     updated_config_path = base_output_dir / "config_after_hpo.yaml"
                     save_config(config, updated_config_path)
                     logging.info(f"Updated config saved to {updated_config_path}")
            else:
                logging.info("No best hyperparameters found or HPO failed. Proceeding with original config.")
        else:
            logging.info("Skipping HPO (disabled in config).")
    else:
        logging.info("Skipping HPO (Optuna not installed).")
    
    # --- Main Comparative Runs --- 
    logging.info(f"Starting comparative analysis for models: {models_to_run} across {num_seeds} seeds.")
    for model_type in models_to_run:
        trained_models_info[model_type] = []
        for seed in range(num_seeds):
            seed_output_dir = base_output_dir / "comparative" / f"model_{model_type}" / f"seed_{seed}"
            metrics = run_single_experiment(seed, model_type, config, seed_output_dir)
            if metrics:
                run_info = {
                    "seed": seed,
                    "model_type": model_type,
                    "metrics": metrics,
                    "output_dir": str(seed_output_dir)
                }
                all_run_metrics.append(run_info)
                # Assume main.py saves the model and we can store its path
                model_path = seed_output_dir / "model.pt" # Assuming this name
                if model_path.exists():
                     trained_models_info[model_type].append(str(model_path))
                else:
                    logging.warning(f"Trained model file not found at {model_path}")
            else:
                logging.error(f"Run failed for seed={seed}, model={model_type}. Skipping.")

    # --- Aggregate and Visualize All Results --- #
    if all_run_metrics:
        logging.info("--- Aggregating and Visualizing All Results --- ")
        results_df = aggregate_results(all_run_metrics, base_output_dir)
        if results_df is not None:
            # Pass the relevant config sections to the visualization function
            visualize_results(results_df, config.get('visualization', {}), config.get('evaluation', {}), base_output_dir)
    else:
        logging.warning("No metrics collected from any runs. Skipping aggregation and visualization.")

    # --- Ablation Studies --- #
    run_ablations_flag = exp_config.get('run_ablations', False)
    if run_ablations_flag:
        run_ablations(config, args.config_path, models_to_run, num_seeds, base_output_dir / "ablations")

    logging.info("Experiment orchestration finished.")

def _perform_paired_t_test(df: pd.DataFrame, model_col: str, metric_col: str, seed_col: str, alpha: float = 0.05):
    """Performs paired t-tests between all pairs of models for a given metric."""
    models = df[model_col].unique()
    if len(models) < 2:
        logging.info(f"Need at least 2 models for paired t-test, found {len(models)}. Skipping for metric '{metric_col}'.")
        return
        
    logging.info(f"--- Paired T-Tests for Metric: {metric_col} --- ")
    
    # Ensure data is sorted consistently by seed within each model group
    df_sorted = df.sort_values(by=[model_col, seed_col])
    
    results = []
    for model1, model2 in itertools.combinations(models, 2):
        try:
            values1 = df_sorted[df_sorted[model_col] == model1][metric_col].values
            values2 = df_sorted[df_sorted[model_col] == model2][metric_col].values
            
            if len(values1) != len(values2):
                logging.warning(f"Unequal number of samples for {model1} ({len(values1)}) and {model2} ({len(values2)}) for metric {metric_col}. Cannot perform paired t-test. Ensure seeds match across runs.")
                continue
            if len(values1) == 0:
                 logging.warning(f"No data found for {model1} or {model2} for metric {metric_col}. Skipping pair.")
                 continue

            # Drop NaNs if present, but pairing must be maintained - T-test handles NaNs
            t_stat, p_value = stats.ttest_rel(values1, values2, nan_policy='omit') # Use omit
            
            significant = p_value < alpha
            mean1 = np.nanmean(values1)
            mean2 = np.nanmean(values2)
            result_str = (
                f"  Comparison: {model1} vs {model2}\n" 
                f"    Means: {mean1:.4f} vs {mean2:.4f}\n" 
                f"    T-statistic: {t_stat:.4f}, P-value: {p_value:.4g}\n" 
                f"    {'SIGNIFICANT difference' if significant else 'NO significant difference'} at alpha={alpha}"
            )
            logging.info(result_str)
            results.append({'model1': model1, 'model2': model2, 'metric': metric_col, 't_stat': t_stat, 'p_value': p_value, 'significant': significant})
            
        except Exception as e:
            logging.error(f"Error during t-test between {model1} and {model2} for {metric_col}: {e}")
            
    logging.info(f"--- End T-Tests for Metric: {metric_col} --- ")
    # Optional: Return results table
    # return pd.DataFrame(results)
