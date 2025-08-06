import matplotlib.pyplot as plt

def plot_comprehensive_results(results_df, folder):
    """
    Plot comprehensive evaluation results including:
    1. Overall and class-specific accuracy
    2. FLOPs and FLOPs reduction
    3. Class-specific F1 scores
    
    Args:
        results_df: DataFrame containing evaluation results
        save_prefix: If provided, plots will be saved with this prefix
    """
    # Create a 2x1 figure layout (accuracy and F1 charts stacked vertically)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 16))
    
    # Plot overall accuracy on the top plot
    ax1.plot(results_df['threshold'], results_df['overall_accuracy'], 
             marker='o', linewidth=2.5, label='Overall Accuracy', color='blue')
    
    # Identify and plot class-specific accuracies
    class_columns = [col for col in results_df.columns if col.endswith('_accuracy') and col != 'overall_accuracy']
    class_colors = {'negative_accuracy': 'red', 'neutral_accuracy': 'green', 'positive_accuracy': 'purple'}
    
    for col in class_columns:
        class_name = col.split('_')[0]
        color = class_colors.get(col, None)  # Use predefined color or None (matplotlib will choose)
        ax1.plot(results_df['threshold'], results_df[col], 
                marker='s', linewidth=2, label=f'{class_name.capitalize()} Accuracy', color=color)
    
    ax1.set_title('Model Accuracy vs Pruning Threshold', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pruning Threshold (%)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(results_df['threshold'])
    ax1.set_ylim([0, 1.05])  # Set y-axis from 0 to just above 1
    ax1.legend(loc='best', fontsize=11)
    
    # Plot class-specific F1 scores on the bottom plot
    ax3.plot(results_df['threshold'], results_df['overall_f1'], 
             marker='o', linewidth=2.5, label='Overall F1', color='blue')
    
    # Plot class-specific F1 scores
    f1_columns = [col for col in results_df.columns if col.endswith('_f1') and col != 'overall_f1']
    class_colors = {'negative_f1': 'red', 'neutral_f1': 'green', 'positive_f1': 'purple'}
    
    for col in f1_columns:
        class_name = col.split('_')[0]
        color = class_colors.get(col, None)
        ax3.plot(results_df['threshold'], results_df[col], 
                marker='s', linewidth=2, label=f'{class_name.capitalize()} F1', color=color)
    
    ax3.set_title('F1 Scores vs Pruning Threshold', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Pruning Threshold (%)', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xticks(results_df['threshold'])
    ax3.set_ylim([0, 1.05])
    ax3.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"runs/{folder}/metrics.png", dpi=300, bbox_inches='tight')
    
    # Figure 2: FLOPs reduction as a separate chart
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    
    # Plot FLOPs reduction percentage
    ax2.plot(results_df['threshold'], results_df['flops_reduction_pct'], 
             marker='o', color='orange', linewidth=2.5, label='FLOPs Reduction %')
    ax2.set_title('FLOPs Reduction vs Pruning Threshold', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Pruning Threshold (%)', fontsize=12)
    ax2.set_ylabel('FLOPs Reduction (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(results_df['threshold'])
    
    plt.tight_layout()
    plt.savefig(f"runs/{folder}/flops_reduction.png", dpi=300, bbox_inches='tight')