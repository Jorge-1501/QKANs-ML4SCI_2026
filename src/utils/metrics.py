import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import seaborn as sns

# Matplotlib parameters for consistent styling
FONT_PARAMS = {'fontsize': 16, 'fontweight': 'bold'}
TICK_PARAMS = {'fontsize': 12}
LEGEND_FONT = {'size': 12}


def plot_loss_history(history, save_path=None):
    """
    Plots the training and validation loss history.
    args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss' lists.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6)) 
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)

    plt.title('Loss History', **FONT_PARAMS)
    plt.xlabel('Epoch', **FONT_PARAMS)
    plt.ylabel('Loss', **FONT_PARAMS)
    plt.xticks(**TICK_PARAMS)
    plt.yticks(**TICK_PARAMS)
    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Loss history fig save in: '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_auc_history(history, save_path=None):
    """
    Plots the training and validation AUC history.
    args:
        history (dict): A dictionary containing 'train_auc' and 'val_auc' lists.
        save_path (str): Path to save the plot image.
    """
    if 'train_auc' not in history or 'val_auc' not in history:
        print(f"Warning: No training or validation AUC data found in history. Skipping AUC plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_auc'], label='Training AUC', color='green', linewidth=2) if history['train_auc'] else None
    plt.plot(history['val_auc'], label='Validation AUC', color='red', linewidth=2) if history['val_auc'] else None

    plt.title('AUC History', **FONT_PARAMS)
    plt.xlabel('Epoch', **FONT_PARAMS)
    plt.ylabel('AUC', **FONT_PARAMS)
    plt.xticks(**TICK_PARAMS)
    plt.yticks(**TICK_PARAMS)
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"AUC history fig save in: '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None):
    """
    Plots the ROC curve.

    Args:
        y_true (array-like): True binary labels.
        y_probs (array-like): Target scores, can either be probability estimates of the positive class.
        save_path(str): Path to save the plot image.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.5f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

    plt.title('Receiver Operating Characteristic (ROC) Curve', **FONT_PARAMS)
    plt.xlabel('False Positive Rate', **FONT_PARAMS)
    plt.ylabel('True Positive Rate', **FONT_PARAMS)
    plt.xticks(**TICK_PARAMS)
    plt.yticks(**TICK_PARAMS)

    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(y_true, y_probs, save_path=None):
    """
    Plots the Precision-Recall curve.
    Args:

    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    
    plt.figure(figsize=(8, 6))
    plt.title('Precisión-Recall Curve', **FONT_PARAMS)
    plt.plot(recall, precision, label=f'Curve PR (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', **FONT_PARAMS)
    plt.ylabel('Precision', **FONT_PARAMS)

    plt.legend(fontsize=LEGEND_FONT['size'])
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-Recall curve plot saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, save_path=None):
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
    
    plt.title('Confusion Matrix', **FONT_PARAMS)
    plt.xlabel('Predicted Value', **FONT_PARAMS)
    plt.ylabel('Real Value', **FONT_PARAMS)
    
    tick_labels = ['bkg (0)', 'top (1)']
    plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, **TICK_PARAMS)
    plt.xticks(ticks=[0.5, 1.5], labels=tick_labels, **TICK_PARAMS)

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()
