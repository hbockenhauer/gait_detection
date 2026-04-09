import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from collections import defaultdict

def plot_roc_curves(results, plots_root_dir, model_name='strokenet'):
    """Plot pooled ROC curves (per dataset) from stored probabilities and labels."""
    grouped = defaultdict(lambda: {'y_true': [], 'probs': []})
    for r in results:
        if 'y_true' not in r or 'probs' not in r:
            continue
        grouped[r['dataset']]['y_true'].append(np.asarray(r['y_true']).astype(int))
        grouped[r['dataset']]['probs'].append(np.asarray(r['probs']).astype(float))

    if len(grouped) == 0:
        print("No probability/label data found for ROC plotting.")
        return

    roc_dir = os.path.join(plots_root_dir, 'ROC', model_name)
    os.makedirs(roc_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))
    any_curve = False

    for dataset, vals in grouped.items():
        y_true = np.concatenate(vals['y_true']) if vals['y_true'] else np.array([])
        probs = np.concatenate(vals['probs']) if vals['probs'] else np.array([])
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            print(f"Skipping ROC for {dataset}: need both classes present.")
            continue

        fpr, tpr, thresholds = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        any_curve = True

        # Youden index gives an operating threshold that balances sensitivity/specificity.
        youden_idx = np.argmax(tpr - fpr)
        best_thr = thresholds[youden_idx]
        print(f"ROC {dataset}: AUC={roc_auc:.3f}, best Youden threshold={best_thr:.3f}")

        plt.plot(fpr, tpr, linewidth=2, label=f"{dataset} (AUC={roc_auc:.3f})")

        roc_points_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})
        roc_points_path = os.path.join(roc_dir, f"roc_points_{dataset.lower()}.csv")
        roc_points_df.to_csv(roc_points_path, index=False)

    if not any_curve:
        plt.close()
        print("No ROC curves were generated.")
        return

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curves by Dataset')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_fig_path = os.path.join(roc_dir, 'roc_curves.png')
    plt.savefig(roc_fig_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC figure: {roc_fig_path}")


def plot_pr_curves(results, plots_root_dir, model_name='strokenet'):
    """Plot pooled precision-recall curves (per dataset) from stored probabilities and labels."""
    grouped = defaultdict(lambda: {'y_true': [], 'probs': []})
    for r in results:
        if 'y_true' not in r or 'probs' not in r:
            continue
        grouped[r['dataset']]['y_true'].append(np.asarray(r['y_true']).astype(int))
        grouped[r['dataset']]['probs'].append(np.asarray(r['probs']).astype(float))

    if len(grouped) == 0:
        print("No probability/label data found for PR plotting.")
        return

    pr_dir = os.path.join(plots_root_dir, 'PR', model_name)
    os.makedirs(pr_dir, exist_ok=True)

    plt.figure(figsize=(8, 7))
    any_curve = False

    for dataset, vals in grouped.items():
        y_true = np.concatenate(vals['y_true']) if vals['y_true'] else np.array([])
        probs = np.concatenate(vals['probs']) if vals['probs'] else np.array([])
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            print(f"Skipping PR for {dataset}: need both classes present.")
            continue

        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        any_curve = True

        # Maximize F1 across thresholds to suggest an operating point.
        if len(thresholds) > 0:
            f1 = 2 * (precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
            best_idx = int(np.argmax(f1))
            best_thr = float(thresholds[best_idx])
            best_f1 = float(f1[best_idx])
            print(f"PR {dataset}: AP={ap:.3f}, best F1={best_f1:.3f} at threshold={best_thr:.3f}")
        else:
            best_thr = np.nan
            best_f1 = np.nan
            print(f"PR {dataset}: AP={ap:.3f}")

        plt.plot(recall, precision, linewidth=2, label=f"{dataset} (AP={ap:.3f})")

        pr_points_df = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'threshold': np.r_[thresholds, np.nan]
        })
        pr_points_path = os.path.join(pr_dir, f"pr_points_{dataset.lower()}.csv")
        pr_points_df.to_csv(pr_points_path, index=False)

    if not any_curve:
        plt.close()
        print("No PR curves were generated.")
        return

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curves by Dataset')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pr_fig_path = os.path.join(pr_dir, 'pr_curves.png')
    plt.savefig(pr_fig_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved PR figure: {pr_fig_path}")