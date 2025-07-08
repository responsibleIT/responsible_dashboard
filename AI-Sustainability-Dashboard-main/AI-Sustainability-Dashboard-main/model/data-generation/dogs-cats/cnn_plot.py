import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is imported if not already

def plot_cnn_comprehensive_results(results_df, folder_path):
    """
    Plot comprehensive evaluation results for CNN pruning.
    Args:
        results_df: DataFrame containing evaluation results.
        folder_path: Path object for the folder to save plots.
    """
    if results_df.empty:
        print("Results DataFrame is empty, skipping plotting.")
        return

    # Ensure folder exists
    folder_path.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Accuracy ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(results_df['threshold'], results_df['overall_accuracy'], 
             marker='o', linewidth=2.5, label='Overall Accuracy', color='blue')
    
    # Class-specific accuracies for cat and dog
    class_colors_acc = {'cat_accuracy': 'red', 'dog_accuracy': 'green'}
    for col_name, color in class_colors_acc.items():
        if col_name in results_df.columns:
            class_label = col_name.split('_')[0].capitalize()
            ax1.plot(results_df['threshold'], results_df[col_name], 
                     marker='s', linewidth=2, label=f'{class_label} Accuracy', color=color)
    
    ax1.set_title('CNN Accuracy vs Pruning Threshold (0-10 Scale)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pruning Threshold (0-10 Scale, where 10 means 100% of max layer weight)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(results_df['threshold']) # Assumes threshold column is 0-10
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(folder_path / "cnn_accuracy_metrics.png", dpi=300)
    plt.show()

    # --- Plot 2: F1 Scores ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(results_df['threshold'], results_df['overall_f1'], 
             marker='o', linewidth=2.5, label='Overall F1', color='blue')
    
    class_colors_f1 = {'cat_f1': 'red', 'dog_f1': 'green'}
    for col_name, color in class_colors_f1.items():
        if col_name in results_df.columns:
            class_label = col_name.split('_')[0].capitalize()
            ax2.plot(results_df['threshold'], results_df[col_name], 
                     marker='s', linewidth=2, label=f'{class_label} F1 Score', color=color)

    ax2.set_title('CNN F1 Scores vs Pruning Threshold (0-10 Scale)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Pruning Threshold (0-10 Scale)', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(results_df['threshold'])
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(folder_path / "cnn_f1_metrics.png", dpi=300)
    plt.show()

    # --- Plot 3: FLOPs Proxy Reduction and Parameter Reduction ---
    fig3, ax3_flops = plt.subplots(figsize=(12, 7))
    
    color_flops = 'tab:orange'
    ax3_flops.set_xlabel('Pruning Threshold (0-10 Scale)', fontsize=12)
    ax3_flops.set_ylabel('FLOPs Reduction Proxy (%)', color=color_flops, fontsize=12)
    ax3_flops.plot(results_df['threshold'], results_df['flops_reduction_pct_proxy'], 
                   marker='o', color=color_flops, linewidth=2.5, label='FLOPs Reduction % (Proxy)')
    ax3_flops.tick_params(axis='y', labelcolor=color_flops)
    ax3_flops.grid(True, linestyle='--', alpha=0.7)
    ax3_flops.set_xticks(results_df['threshold'])

    ax3_params = ax3_flops.twinx()  # instantiate a second axes that shares the same x-axis
    color_params = 'tab:purple'
    ax3_params.set_ylabel('Parameter Reduction (%)', color=color_params, fontsize=12)
    ax3_params.plot(results_df['threshold'], results_df['params_reduction_pct'], 
                    marker='x', color=color_params, linewidth=2.5, linestyle='--', label='Parameter Reduction %')
    ax3_params.tick_params(axis='y', labelcolor=color_params)

    fig3.suptitle('CNN Resource Reduction vs Pruning Threshold (0-10 Scale)', fontsize=14, fontweight='bold')
    fig3.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax3_flops.transAxes)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(folder_path / "cnn_resource_reduction.png", dpi=300)
    plt.show()