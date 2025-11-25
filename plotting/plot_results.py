import json
import numpy as np
from matplotlib import pyplot as plt

def plot_results(results_dir, results_data):
    """
    Plot results from experiments.
    
    Args:
        results_dir (str): Directory to save plots
        results_data (dict): Results data containing raw results for each method.
    """
    
    # Extract data from results
    methods = [method['name'] for method in results_data['methods']]
    
    # Compute statistics from raw results
    train_regrets_mean = []
    train_regrets_ste = []
    val_regrets_mean = []
    val_regrets_ste = []
    test_regrets_mean = []
    test_regrets_ste = []
    train_times_mean = []
    train_times_ste = []

    for method in results_data['methods']:
        # For training and validation regrets, we need to handle lists of different lengths
        # First get max length of training and validation sequences
        train_regrets = method['train_regrets']
        val_regrets = method['val_regrets']
        max_train_len = max(len(x) for x in train_regrets)
        max_val_len = max(len(x) for x in val_regrets)
        
        # Extend shorter lists with their final element
        train_regrets_padded = [x + [x[-1]] * (max_train_len - len(x)) for x in train_regrets]
        val_regrets_padded = [x + [x[-1]] * (max_val_len - len(x)) for x in val_regrets]
        
        # Convert to numpy arrays for statistics
        train_regrets_arr = np.array(train_regrets_padded)
        val_regrets_arr = np.array(val_regrets_padded)
        test_regrets_arr = np.array(method['test_regrets'])
        train_times_arr = np.array(method['train_times'])
        
        # Compute means and standard errors
        train_regrets_mean.append(np.mean(train_regrets_arr, axis=0))
        train_regrets_ste.append(np.std(train_regrets_arr, axis=0) / np.sqrt(train_regrets_arr.shape[0]))
        val_regrets_mean.append(np.mean(val_regrets_arr, axis=0))
        val_regrets_ste.append(np.std(val_regrets_arr, axis=0) / np.sqrt(val_regrets_arr.shape[0]))
        test_regrets_mean.append(np.mean(test_regrets_arr))
        test_regrets_ste.append(np.std(test_regrets_arr) / np.sqrt(len(test_regrets_arr)))
        train_times_mean.append(np.mean(train_times_arr))
        train_times_ste.append(np.std(train_times_arr) / np.sqrt(len(train_times_arr)))


    log_every_n_epochs = results_data['parameters'].get('log_every_n_epochs', 1)

    # Set up plot style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
    colors = colors[:len(methods)]
    width = 0.35

    # Learning curves (training and validation)
    plt.figure(figsize=(7, 4.5))
    plt.yscale("log")
    for i, method_name in enumerate(methods):
        epochs = np.arange(0, len(train_regrets_mean[i])) * log_every_n_epochs
        plt.plot(epochs, train_regrets_mean[i], color=colors[i], label=method_name)
        plt.fill_between(epochs,
                        train_regrets_mean[i] - train_regrets_ste[i],
                        train_regrets_mean[i] + train_regrets_ste[i],
                        alpha=0.2, color=colors[i])

    plt.xlabel("Epoch")
    plt.ylabel("Training Regret")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{results_dir}/training_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Validation curves
    plt.figure(figsize=(7, 4.5))
    plt.yscale("log")
    for i, method_name in enumerate(methods):
        epochs = np.arange(0, len(val_regrets_mean[i])) * log_every_n_epochs
        plt.plot(epochs, val_regrets_mean[i], color=colors[i], label=method_name)
        plt.fill_between(epochs,
                        val_regrets_mean[i] - val_regrets_ste[i],
                        val_regrets_mean[i] + val_regrets_ste[i],
                        alpha=0.2, color=colors[i])

    plt.xlabel("Epoch")
    plt.ylabel("Validation Regret")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{results_dir}/validation_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Test regret bar plot
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(methods))
    plt.bar(x, test_regrets_mean, width, yerr=test_regrets_ste, capsize=5)
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.ylabel('Test Regret')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/test_regret.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Training time bar plot
    plt.figure(figsize=(7, 4.5))
    plt.yscale("log")
    plt.bar(x, train_times_mean, width, yerr=train_times_ste, capsize=5)
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#ff7f0e', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    marker_styles = ['o', 'D', '^', 's', 'X', '*']

    fig, ax = plt.subplots(figsize=(3, 3))
    for i, method in enumerate(methods):
        ax.scatter(train_times_mean[i], test_regrets_mean[i],
                   c=colors[i % len(colors)], s=115, edgecolor='black', linewidth=1.2,
                   marker=marker_styles[i % len(marker_styles)], label=method, alpha=0.8)

    ax.set_xlabel("Training Time (s)", fontsize=12)
    ax.set_ylabel("Test Regret", fontsize=12)
    ax.set_yscale('log')

    ax.grid(visible=True, linestyle='--', alpha=0.5)
    ax.legend(
        title="",
        loc='upper center',
        bbox_to_anchor=(0.5, 1.4),
        ncol=3,
        frameon=False,
        fontsize=14,
        markerscale=2,
        columnspacing=2.0,
        labelspacing=1.2,
    )

    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(f"{results_dir}/time_regret_tradeoff.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    results_path = "" # Path to results file
    results_file = results_path + "/results.json"
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    plot_results(results_path, results_data)
