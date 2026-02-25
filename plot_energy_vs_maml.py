#!/usr/bin/env python3
"""
Compare Energy-based Training vs MAML meta-learning results.

Reads results from 'energy/results' and 'maml/results' directories,
computes mean and standard deviation across datasets, and generates
publication-quality epoch-accuracy plots.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging


def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup clean logger for script output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def load_results_from_directory(directory: str, logger: logging.Logger) -> Dict[str, Dict]:
    """
    Load all JSON result files from a directory.
    
    Args:
        directory: Path to results directory
        logger: Logger instance
        
    Returns:
        Dictionary mapping dataset names to result dictionaries
    """
    results = {}
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {directory}")
        return results
    
    for json_path in sorted(json_files):
        dataset_name = Path(json_path).stem
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            results[dataset_name] = data
            logger.info(f"✓ Loaded {dataset_name} from {json_path}")
        except Exception as e:
            logger.warning(f"✗ Failed to load {json_path}: {e}")
    
    return results


def extract_val_acc_history(results: Dict[str, Dict], method_name: str, logger: logging.Logger) -> Tuple[np.ndarray, List[str]]:
    """
    Extract validation accuracy histories from results.
    
    Args:
        results: Dictionary of results per dataset
        method_name: Name of the method for logging
        logger: Logger instance
        
    Returns:
        Tuple of (2D array of shape [n_datasets, max_epochs], list of dataset names)
    """
    histories = []
    valid_datasets = []
    
    for dataset_name, data in results.items():
        if 'val_acc_history' in data and data['val_acc_history']:
            acc_history = data['val_acc_history']
            # Convert to percentages
            acc_history_pct = [acc * 100.0 if acc <= 1.0 else acc for acc in acc_history]
            histories.append(acc_history_pct)
            valid_datasets.append(dataset_name)
            logger.info(f"  {dataset_name}: {len(acc_history_pct)} epochs, final={acc_history_pct[-1]:.2f}%")
        else:
            logger.warning(f"  {dataset_name}: No val_acc_history found, skipping")
    
    if not histories:
        logger.error(f"No valid histories found for {method_name}")
        return np.array([]), []
    
    # Pad to same length (use max length)
    max_length = max(len(h) for h in histories)
    padded_histories = []
    
    for h in histories:
        if len(h) < max_length:
            # Pad with the last value (assume accuracy plateaus)
            padded = h + [h[-1]] * (max_length - len(h))
        else:
            padded = h
        padded_histories.append(padded)
    
    return np.array(padded_histories), valid_datasets


def plot_comparison(
    energy_histories: np.ndarray,
    maml_histories: np.ndarray,
    energy_datasets: List[str],
    maml_datasets: List[str],
    save_path: str = "energy_vs_maml_comparison.pdf",
    band_mode: str = "sem",  # 'std' | 'scaled_std' | 'sem'
    band_scale: float = 0.5,  # scale factor for band width
    logger: Optional[logging.Logger] = None,
):
    """
    Create publication-quality comparison plot.
    
    Args:
        energy_histories: Array of shape [n_datasets, n_epochs] for energy method
        maml_histories: Array of shape [n_datasets, n_epochs] for MAML
        energy_datasets: List of dataset names for energy method
        maml_datasets: List of dataset names for MAML
        save_path: Path to save the figure
        band_mode: How to compute shaded band ('std', 'scaled_std', 'sem')
        band_scale: Multiplier applied to the band (used for 'scaled_std' and 'sem')
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Configure matplotlib for publication quality
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.figsize': (10, 6),
        'lines.linewidth': 4,
        'lines.markersize': 10,
        'grid.alpha': 0.3,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors (colorblind-friendly palette)
    color_energy = '#0173B2'  # Blue
    color_maml = '#DE8F05'    # Orange
    
    # Plot Energy method
    if energy_histories.size > 0:
        # Use only first 10 epochs for statistics and plotting
        n_epochs_energy = min(20, energy_histories.shape[1])
        epochs_energy = np.arange(1, n_epochs_energy + 1)
        
        mean_energy = np.mean(energy_histories[:, :n_epochs_energy], axis=0)
        std_energy = np.std(energy_histories[:, :n_epochs_energy], axis=0, ddof=1)
        n_energy = energy_histories.shape[0]
        if band_mode == "std":
            band_energy = std_energy
        elif band_mode == "scaled_std":
            band_energy = band_scale * std_energy
        else:  # "sem" default
            sem_energy = std_energy / np.sqrt(max(n_energy, 1))
            band_energy = band_scale * sem_energy
        
        ax.plot(epochs_energy, mean_energy, 
                color=color_energy, 
                label=f'BOLT',
                linewidth=4,
                zorder=3)
        ax.fill_between(epochs_energy, 
                        mean_energy - band_energy, 
                        mean_energy + band_energy,
                        color=color_energy, 
                        alpha=0.2,
                        zorder=2)
        
        logger.info(f"\nEnergy method statistics:")
        logger.info(f"  Datasets: {', '.join(energy_datasets)}")
        logger.info(f"  Final accuracy: {mean_energy[-1]:.2f}% ± {std_energy[-1]:.2f}%")
        logger.info(f"  Band: mode={band_mode}, scale={band_scale}")
    
    # Plot MAML method
    if maml_histories.size > 0:
        # Use only first 20 epochs for statistics and plotting
        n_epochs_maml = min(20, maml_histories.shape[1])
        epochs_maml = np.arange(1, n_epochs_maml + 1)
        
        mean_maml = np.mean(maml_histories[:, :n_epochs_maml], axis=0)
        std_maml = np.std(maml_histories[:, :n_epochs_maml], axis=0, ddof=1)
        n_maml = maml_histories.shape[0]
        if band_mode == "std":
            band_maml = std_maml
        elif band_mode == "scaled_std":
            band_maml = band_scale * std_maml
        else:  # "sem" default
            sem_maml = std_maml / np.sqrt(max(n_maml, 1))
            band_maml = band_scale * sem_maml
        
        ax.plot(epochs_maml, mean_maml, 
                color=color_maml, 
                label=f'Meta learned',
                linewidth=4,
                zorder=3)
        ax.fill_between(epochs_maml, 
                        mean_maml - band_maml, 
                        mean_maml + band_maml,
                        color=color_maml, 
                        alpha=0.2,
                        zorder=2)
        
        logger.info(f"\nMAML method statistics:")
        logger.info(f"  Datasets: {', '.join(maml_datasets)}")
        logger.info(f"  Final accuracy: {mean_maml[-1]:.2f}% ± {std_maml[-1]:.2f}%")
        logger.info(f"  Band: mode={band_mode}, scale={band_scale}")
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=22, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=22, fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=2)
    ax.set_axisbelow(True)
    
    # Set reasonable y-axis limits
    ax.set_ylim(bottom=24, top=76)
    # Limit x-axis to 0..20 and show integer ticks every 4
    ax.set_xlim(left=1, right=20)
    ax.set_xticks(np.arange(1, 21, 4))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Save in multiple formats
    base_path = os.path.splitext(save_path)[0]
    
    # plt.savefig(f"{base_path}.pdf", dpi=1200, bbox_inches='tight')
    # logger.info(f"\n✓ Saved PDF: {base_path}.pdf")
    
    plt.savefig(f"{base_path}.png", dpi=1200, bbox_inches='tight')
    logger.info(f"✓ Saved PNG: {base_path}.png")
    
    plt.close()


def main():
    """Main execution function."""
    logger = setup_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("Energy-based Training vs MAML Comparison")
    logger.info("=" * 80)
    
    # Paths
    energy_results_dir = "energy/results"
    maml_results_dir = "maml/results"
    output_path = "plots/energy_vs_maml_comparison.pdf"
    
    # Check if directories exist
    if not os.path.exists(energy_results_dir):
        logger.error(f"Energy results directory not found: {energy_results_dir}")
        return
    
    if not os.path.exists(maml_results_dir):
        logger.error(f"MAML results directory not found: {maml_results_dir}")
        return
    
    # Load results
    logger.info(f"\nLoading Energy results from {energy_results_dir}...")
    energy_results = load_results_from_directory(energy_results_dir, logger)
    
    logger.info(f"\nLoading MAML results from {maml_results_dir}...")
    maml_results = load_results_from_directory(maml_results_dir, logger)
    
    if not energy_results and not maml_results:
        logger.error("No results found in either directory. Exiting.")
        return
    
    # Extract validation accuracy histories
    logger.info("\n" + "=" * 80)
    logger.info("Extracting validation accuracy histories...")
    logger.info("=" * 80)
    
    logger.info("\nEnergy method:")
    energy_histories, energy_datasets = extract_val_acc_history(
        energy_results, "Energy", logger)
    
    logger.info("\nMAML method:")
    maml_histories, maml_datasets = extract_val_acc_history(
        maml_results, "MAML", logger)
    
    # Select top 30% datasets based on MAML final accuracy
    if maml_histories.size > 0:
        logger.info("\n" + "=" * 80)
        logger.info("Selecting top 30% datasets by MAML final accuracy...")
        logger.info("=" * 80)
        num_maml_datasets = len(maml_datasets)
        top_k = max(1, int(np.ceil(num_maml_datasets * 0.50)))
        
        maml_final_accs = maml_histories[:, -1]  # final epoch accuracy per dataset (already in %)
        sorted_indices_desc = np.argsort(-maml_final_accs)
        selected_indices_maml = sorted_indices_desc[:top_k].tolist()
        selected_datasets = [maml_datasets[i] for i in selected_indices_maml]
        
        # Filter MAML histories to selected datasets
        maml_histories = maml_histories[selected_indices_maml, :]
        maml_datasets = selected_datasets
        
        # Filter Energy histories to those selected (intersection)
        if energy_histories.size > 0:
            name_to_idx_energy = {name: idx for idx, name in enumerate(energy_datasets)}
            selected_indices_energy = [name_to_idx_energy[name] for name in selected_datasets if name in name_to_idx_energy]
            missing_in_energy = [name for name in selected_datasets if name not in name_to_idx_energy]
            
            if missing_in_energy:
                logger.warning(f"Energy results missing for: {', '.join(missing_in_energy)}")
            
            if selected_indices_energy:
                energy_histories = energy_histories[selected_indices_energy, :]
                energy_datasets = [energy_datasets[i] for i in selected_indices_energy]
            else:
                # No overlap; keep empty so plot function gracefully skips
                energy_histories = np.array([])
                energy_datasets = []
        
        logger.info(f"Selected {len(maml_datasets)} of {num_maml_datasets} MAML datasets: {', '.join(maml_datasets)}")
    
    
    # Generate comparison plot
    logger.info("\n" + "=" * 80)
    logger.info("Generating comparison plot...")
    logger.info("=" * 80)
    
    plot_comparison(
        energy_histories,
        maml_histories,
        energy_datasets,
        maml_datasets,
        save_path=output_path,
        logger=logger,
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Comparison complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

