'''
Script to analyse whether gait detection is better with one wrist or both wrists.
Evaluates affected wrist, unaffected wrist, and both wrists for QSense Clinic patients.
'''
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.StrokeNet.strokenet_utils import (
    WEIGHTS_PATH,
    load_finetuned_model,
    plot_subject_timeline,
    evaluate_qsense_dataset,
    _load_qsense_file,
    extract_windows_with_gaps_and_activity,
    run_inference,
    compute_metrics,
    get_discontinuity_times,
    CONF_THRESH,
)
from config.paths import PLOTS_DIR, RESULTS_DIR, QSENSE_CLINIC
import utils.plot_ROC_PR as plot_ROC_PR

# Affected side per patient in QSense Clinic [sub1, sub2, sub3...]
affected_wrist_patient = ['RW', 'LW', 'LW', 'LW', 'LW']


def find_optimal_thresholds(y_true, probs, beta=1.0):
    """
    Compute threshold candidates from ROC and PR curves.
    Returns best Youden J threshold (ROC) and best F-beta threshold (PR).
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None

    result = {}

    # ROC: maximize Youden's J = TPR - FPR.
    fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
    finite = np.isfinite(roc_thresholds)
    if np.any(finite):
        fpr_f = fpr[finite]
        tpr_f = tpr[finite]
        roc_thr_f = roc_thresholds[finite]
        j_vals = tpr_f - fpr_f
        best_j_idx = int(np.argmax(j_vals))
        result['youden_j'] = {
            'threshold': float(roc_thr_f[best_j_idx]),
            'j_score': float(j_vals[best_j_idx]),
            'fpr': float(fpr_f[best_j_idx]),
            'tpr': float(tpr_f[best_j_idx]),
        }

    # PR: maximize F-beta.
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
    if len(pr_thresholds) > 0:
        p = precision[:-1]
        r = recall[:-1]
        denom = (beta ** 2) * p + r
        fbeta = np.where(denom > 0, (1 + beta ** 2) * p * r / denom, 0.0)
        best_f_idx = int(np.argmax(fbeta))
        result['f_beta'] = {
            'threshold': float(pr_thresholds[best_f_idx]),
            'f_beta': float(fbeta[best_f_idx]),
            'beta': float(beta),
            'precision': float(p[best_f_idx]),
            'recall': float(r[best_f_idx]),
        }

    return result if result else None


def get_wrist_file(folder_path, wrist_type):
    """Get the file path for a specific wrist. Returns None if not found."""
    if wrist_type == 'right':
        fname = 's1_1RW.txt'
    elif wrist_type == 'left':
        fname = 's2_2LW.txt'
    else:
        return None
    
    fpath = os.path.join(folder_path, fname)
    return fpath if os.path.exists(fpath) else None


def load_wrist_data(folder_path, wrist_type, folder_name):
    """Load acceleration data for a specific wrist."""
    fpath = get_wrist_file(folder_path, wrist_type)
    if fpath is None:
        return None
    
    try:
        loaded = _load_qsense_file(fpath, folder_name)
        # Backward/forward compatible with helper returning either 4 or 5 values.
        if len(loaded) == 5:
            times, acc, y_binary, activities, _ = loaded
        else:
            times, acc, y_binary, activities = loaded
        return times, acc, y_binary, activities
    except Exception as e:
        print(f"    Error loading {wrist_type} wrist: {e}")
        return None


def combine_wrist_data(
    times_rw,
    acc_rw,
    y_binary_rw,
    activities_rw,
    times_lw,
    acc_lw,
    y_binary_lw,
    activities_lw,
    time_tolerance=0.02,
):
    """
    Combine both wrists at raw-sample level using timestamp alignment.
    For matched samples, average acceleration; for unmatched samples, keep single wrist data.
    """
    i, j = 0, 0
    n_rw, n_lw = len(times_rw), len(times_lw)

    times_combined = []
    acc_combined = []
    y_binary_combined = []
    activities_combined = []

    while i < n_rw and j < n_lw:
        t_rw = times_rw[i]
        t_lw = times_lw[j]
        dt = t_rw - t_lw

        if abs(dt) <= time_tolerance:
            times_combined.append((t_rw + t_lw) / 2.0)
            acc_combined.append((acc_rw[i] + acc_lw[j]) / 2.0)
            y_binary_combined.append(max(y_binary_rw[i], y_binary_lw[j]))
            activities_combined.append(activities_rw[i])
            i += 1
            j += 1
        elif dt < 0:
            times_combined.append(t_rw)
            acc_combined.append(acc_rw[i])
            y_binary_combined.append(y_binary_rw[i])
            activities_combined.append(activities_rw[i])
            i += 1
        else:
            times_combined.append(t_lw)
            acc_combined.append(acc_lw[j])
            y_binary_combined.append(y_binary_lw[j])
            activities_combined.append(activities_lw[j])
            j += 1

    while i < n_rw:
        times_combined.append(times_rw[i])
        acc_combined.append(acc_rw[i])
        y_binary_combined.append(y_binary_rw[i])
        activities_combined.append(activities_rw[i])
        i += 1

    while j < n_lw:
        times_combined.append(times_lw[j])
        acc_combined.append(acc_lw[j])
        y_binary_combined.append(y_binary_lw[j])
        activities_combined.append(activities_lw[j])
        j += 1

    if len(times_combined) < 2:
        return None

    return (
        np.array(times_combined),
        np.array(acc_combined),
        np.array(y_binary_combined),
        np.array(activities_combined),
    )


def evaluate_specific_wrist(model, device, folder_path, folder_name, wrist_type):
    """Evaluate gait detection for a specific wrist."""
    wrist_data = load_wrist_data(folder_path, wrist_type, folder_name)
    
    if wrist_data is None:
        return None
    
    times, acc, y_binary, activities = wrist_data
    discontinuity_times = get_discontinuity_times(times)
    
    wins_np, y_true, win_times, win_activities = extract_windows_with_gaps_and_activity(
        times, acc, y_binary, activities
    )
    
    if wins_np is None:
        return None
    
    probs = run_inference(model, wins_np, device)
    y_pred = (probs > CONF_THRESH).astype(int)
    prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
    
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'accuracy': acc_score,
        'confusion_matrix': cm,
        'probs': probs,
        'y_true': y_true,
        'y_pred': y_pred,
        'win_times': win_times,
        'win_activities': win_activities,
        'discontinuity_times': discontinuity_times,
    }


def evaluate_both_wrists(model, device, folder_path, folder_name):
    """Evaluate gait detection using both wrists fused at acceleration level."""
    # Load both wrists
    rw_data = load_wrist_data(folder_path, 'right', folder_name)
    lw_data = load_wrist_data(folder_path, 'left', folder_name)
    
    if rw_data is None and lw_data is None:
        return None

    # If one wrist is unavailable, fall back to the available wrist
    if rw_data is None:
        times_lw, acc_lw, y_binary_lw, activities_lw = lw_data
        discontinuity_times = get_discontinuity_times(times_lw)
        wins_np, y_true, win_times, win_activities = extract_windows_with_gaps_and_activity(
            times_lw, acc_lw, y_binary_lw, activities_lw
        )
        if wins_np is None:
            return None
        probs = run_inference(model, wins_np, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
        return {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': acc_score,
            'confusion_matrix': cm,
            'probs': probs,
            'y_true': y_true,
            'y_pred': y_pred,
            'win_times': win_times,
            'win_activities': win_activities,
            'discontinuity_times': discontinuity_times,
        }

    if lw_data is None:
        times_rw, acc_rw, y_binary_rw, activities_rw = rw_data
        discontinuity_times = get_discontinuity_times(times_rw)
        wins_np, y_true, win_times, win_activities = extract_windows_with_gaps_and_activity(
            times_rw, acc_rw, y_binary_rw, activities_rw
        )
        if wins_np is None:
            return None
        probs = run_inference(model, wins_np, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
        return {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': acc_score,
            'confusion_matrix': cm,
            'probs': probs,
            'y_true': y_true,
            'y_pred': y_pred,
            'win_times': win_times,
            'win_activities': win_activities,
            'discontinuity_times': discontinuity_times,
        }
    
    times_rw, acc_rw, y_binary_rw, activities_rw = rw_data
    times_lw, acc_lw, y_binary_lw, activities_lw = lw_data

    combined = combine_wrist_data(
        times_rw,
        acc_rw,
        y_binary_rw,
        activities_rw,
        times_lw,
        acc_lw,
        y_binary_lw,
        activities_lw,
    )

    if combined is None:
        return None

    times_fused, acc_fused, y_binary_fused, activities_fused = combined
    discontinuity_times = get_discontinuity_times(times_fused)

    wins_np, y_true, win_times, win_activities = extract_windows_with_gaps_and_activity(
        times_fused, acc_fused, y_binary_fused, activities_fused
    )

    if wins_np is None:
        return None

    probs = run_inference(model, wins_np, device)
    y_pred = (probs > CONF_THRESH).astype(int)

    prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)

    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'accuracy': acc_score,
        'confusion_matrix': cm,
        'probs': probs,
        'y_true': y_true,
        'y_pred': y_pred,
        'win_times': win_times,
        'win_activities': win_activities,
        'discontinuity_times': discontinuity_times,
    }


def match_windows_by_time(win_times_1, win_times_2, time_tolerance=0.5):
    """
    Match windows from two wrists based on timing.
    Returns lists of indices for matching windows.
    """
    matches = []  # [(idx_1, idx_2), ...]
    unmatched_1 = list(range(len(win_times_1)))
    unmatched_2 = list(range(len(win_times_2)))
    
    for i, t1 in enumerate(win_times_1):
        for j, t2 in enumerate(win_times_2):
            if abs(t1 - t2) < time_tolerance:
                matches.append((i, j))
                if i in unmatched_1:
                    unmatched_1.remove(i)
                if j in unmatched_2:
                    unmatched_2.remove(j)
                break
    
    return matches, unmatched_1, unmatched_2


def evaluate_both_wrists_prob_average(model, device, folder_path, folder_name):
    """
    Evaluate both wrists by averaging probabilities.
    Run inference separately on each wrist, then average or use single if one wrist missing data.
    """
    # Load both wrists and extract windows separately (full length, no trimming)
    rw_data = load_wrist_data(folder_path, 'right', folder_name)
    lw_data = load_wrist_data(folder_path, 'left', folder_name)
    
    if rw_data is None or lw_data is None:
        return None
    
    times_rw, acc_rw, y_binary_rw, activities_rw = rw_data
    times_lw, acc_lw, y_binary_lw, activities_lw = lw_data
    
    # Extract windows from both wrists (full data, no trimming)
    wins_rw, y_true_rw, win_times_rw, win_activities_rw = extract_windows_with_gaps_and_activity(
        times_rw, acc_rw, y_binary_rw, activities_rw
    )
    
    wins_lw, y_true_lw, win_times_lw, win_activities_lw = extract_windows_with_gaps_and_activity(
        times_lw, acc_lw, y_binary_lw, activities_lw
    )
    
    if wins_rw is None and wins_lw is None:
        return None
    
    # If only one wrist has data, use that
    if wins_rw is None:
        probs = run_inference(model, wins_lw, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        discontinuity_times = get_discontinuity_times(times_lw)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true_lw, y_pred)
        return {
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
            'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_lw, 'y_pred': y_pred,
            'win_times': win_times_lw, 'win_activities': win_activities_lw,
            'discontinuity_times': discontinuity_times,
        }
    
    if wins_lw is None:
        probs = run_inference(model, wins_rw, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        discontinuity_times = get_discontinuity_times(times_rw)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true_rw, y_pred)
        return {
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
            'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_rw, 'y_pred': y_pred,
            'win_times': win_times_rw, 'win_activities': win_activities_rw,
            'discontinuity_times': discontinuity_times,
        }
    
    # Match windows by time
    matches, unmatched_rw, unmatched_lw = match_windows_by_time(win_times_rw, win_times_lw)
    
    # Build combined results
    probs_combined = []
    y_true_combined = []
    win_times_combined = []
    win_activities_combined = []
    
    # Process matched windows (average probabilities)
    probs_rw = run_inference(model, wins_rw, device)
    probs_lw = run_inference(model, wins_lw, device)
    
    for idx_rw, idx_lw in matches:
        probs_combined.append((probs_rw[idx_rw] + probs_lw[idx_lw]) / 2.0)
        y_true_combined.append(y_true_rw[idx_rw])
        win_times_combined.append(win_times_rw[idx_rw])
        win_activities_combined.append(win_activities_rw[idx_rw])
    
    # Process unmatched right wrist windows
    for idx_rw in unmatched_rw:
        probs_combined.append(probs_rw[idx_rw])
        y_true_combined.append(y_true_rw[idx_rw])
        win_times_combined.append(win_times_rw[idx_rw])
        win_activities_combined.append(win_activities_rw[idx_rw])
    
    # Process unmatched left wrist windows
    for idx_lw in unmatched_lw:
        probs_combined.append(probs_lw[idx_lw])
        y_true_combined.append(y_true_lw[idx_lw])
        win_times_combined.append(win_times_lw[idx_lw])
        win_activities_combined.append(win_activities_lw[idx_lw])
    
    if len(probs_combined) == 0:
        return None
    
    probs = np.array(probs_combined)
    y_true = np.array(y_true_combined)
    y_pred = (probs > CONF_THRESH).astype(int)
    discontinuity_times = get_discontinuity_times(times_rw)
    
    prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
    
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'accuracy': acc_score,
        'confusion_matrix': cm,
        'probs': probs,
        'y_true': y_true,
        'y_pred': y_pred,
        'win_times': np.array(win_times_combined),
        'win_activities': np.array(win_activities_combined),
        'discontinuity_times': discontinuity_times,
    }


def evaluate_both_wrists_prob_max(model, device, folder_path, folder_name):
    """
    Evaluate both wrists by taking the maximum probability.
    Run inference separately on each wrist, use the higher confidence prediction.
    Fall back to single wrist if one is missing data.
    """
    rw_data = load_wrist_data(folder_path, 'right', folder_name)
    lw_data = load_wrist_data(folder_path, 'left', folder_name)
    
    if rw_data is None or lw_data is None:
        return None
    
    times_rw, acc_rw, y_binary_rw, activities_rw = rw_data
    times_lw, acc_lw, y_binary_lw, activities_lw = lw_data
    
    # Extract windows from both wrists (full data, no trimming)
    wins_rw, y_true_rw, win_times_rw, win_activities_rw = extract_windows_with_gaps_and_activity(
        times_rw, acc_rw, y_binary_rw, activities_rw
    )
    
    wins_lw, y_true_lw, win_times_lw, win_activities_lw = extract_windows_with_gaps_and_activity(
        times_lw, acc_lw, y_binary_lw, activities_lw
    )
    
    if wins_rw is None and wins_lw is None:
        return None
    
    # If only one wrist has data, use that
    if wins_rw is None:
        probs = run_inference(model, wins_lw, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        discontinuity_times = get_discontinuity_times(times_lw)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true_lw, y_pred)
        return {
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
            'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_lw, 'y_pred': y_pred,
            'win_times': win_times_lw, 'win_activities': win_activities_lw,
            'discontinuity_times': discontinuity_times,
        }
    
    if wins_lw is None:
        probs = run_inference(model, wins_rw, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        discontinuity_times = get_discontinuity_times(times_rw)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true_rw, y_pred)
        return {
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
            'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_rw, 'y_pred': y_pred,
            'win_times': win_times_rw, 'win_activities': win_activities_rw,
            'discontinuity_times': discontinuity_times,
        }
    
    # Match windows by time
    matches, unmatched_rw, unmatched_lw = match_windows_by_time(win_times_rw, win_times_lw)
    
    # Build combined results
    probs_combined = []
    y_true_combined = []
    win_times_combined = []
    win_activities_combined = []
    
    # Run inference
    probs_rw = run_inference(model, wins_rw, device)
    probs_lw = run_inference(model, wins_lw, device)
    
    # Process matched windows (take maximum)
    for idx_rw, idx_lw in matches:
        probs_combined.append(np.maximum(probs_rw[idx_rw], probs_lw[idx_lw]))
        y_true_combined.append(y_true_rw[idx_rw])
        win_times_combined.append(win_times_rw[idx_rw])
        win_activities_combined.append(win_activities_rw[idx_rw])
    
    # Process unmatched right wrist windows
    for idx_rw in unmatched_rw:
        probs_combined.append(probs_rw[idx_rw])
        y_true_combined.append(y_true_rw[idx_rw])
        win_times_combined.append(win_times_rw[idx_rw])
        win_activities_combined.append(win_activities_rw[idx_rw])
    
    # Process unmatched left wrist windows
    for idx_lw in unmatched_lw:
        probs_combined.append(probs_lw[idx_lw])
        y_true_combined.append(y_true_lw[idx_lw])
        win_times_combined.append(win_times_lw[idx_lw])
        win_activities_combined.append(win_activities_lw[idx_lw])
    
    if len(probs_combined) == 0:
        return None
    
    probs = np.array(probs_combined)
    y_true = np.array(y_true_combined)
    y_pred = (probs > CONF_THRESH).astype(int)
    discontinuity_times = get_discontinuity_times(times_rw)
    
    prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
    
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'accuracy': acc_score,
        'confusion_matrix': cm,
        'probs': probs,
        'y_true': y_true,
        'y_pred': y_pred,
        'win_times': np.array(win_times_combined),
        'win_activities': np.array(win_activities_combined),
        'discontinuity_times': discontinuity_times,
    }


def evaluate_both_wrists_prob_min(model, device, folder_path, folder_name):
    """
    Evaluate both wrists by taking the minimum probability (consensus).
    Run inference separately on each wrist, use the more conservative prediction.
    Fall back to single wrist if one is missing data.
    """
    rw_data = load_wrist_data(folder_path, 'right', folder_name)
    lw_data = load_wrist_data(folder_path, 'left', folder_name)
    
    if rw_data is None or lw_data is None:
        return None
    
    times_rw, acc_rw, y_binary_rw, activities_rw = rw_data
    times_lw, acc_lw, y_binary_lw, activities_lw = lw_data
    
    # Extract windows from both wrists (full data, no trimming)
    wins_rw, y_true_rw, win_times_rw, win_activities_rw = extract_windows_with_gaps_and_activity(
        times_rw, acc_rw, y_binary_rw, activities_rw
    )
    
    wins_lw, y_true_lw, win_times_lw, win_activities_lw = extract_windows_with_gaps_and_activity(
        times_lw, acc_lw, y_binary_lw, activities_lw
    )
    
    if wins_rw is None and wins_lw is None:
        return None
    
    # If only one wrist has data, use that
    if wins_rw is None:
        probs = run_inference(model, wins_lw, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        discontinuity_times = get_discontinuity_times(times_lw)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true_lw, y_pred)
        return {
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
            'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_lw, 'y_pred': y_pred,
            'win_times': win_times_lw, 'win_activities': win_activities_lw,
            'discontinuity_times': discontinuity_times,
        }
    
    if wins_lw is None:
        probs = run_inference(model, wins_rw, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        discontinuity_times = get_discontinuity_times(times_rw)
        prec, rec, f1, acc_score, cm = compute_metrics(y_true_rw, y_pred)
        return {
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
            'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_rw, 'y_pred': y_pred,
            'win_times': win_times_rw, 'win_activities': win_activities_rw,
            'discontinuity_times': discontinuity_times,
        }
    
    # Match windows by time
    matches, unmatched_rw, unmatched_lw = match_windows_by_time(win_times_rw, win_times_lw)
    
    # Build combined results
    probs_combined = []
    y_true_combined = []
    win_times_combined = []
    win_activities_combined = []
    
    # Run inference
    probs_rw = run_inference(model, wins_rw, device)
    probs_lw = run_inference(model, wins_lw, device)
    
    # Process matched windows (take minimum for consensus)
    for idx_rw, idx_lw in matches:
        probs_combined.append(np.minimum(probs_rw[idx_rw], probs_lw[idx_lw]))
        y_true_combined.append(y_true_rw[idx_rw])
        win_times_combined.append(win_times_rw[idx_rw])
        win_activities_combined.append(win_activities_rw[idx_rw])
    
    # Process unmatched right wrist windows (use single prediction)
    for idx_rw in unmatched_rw:
        probs_combined.append(probs_rw[idx_rw])
        y_true_combined.append(y_true_rw[idx_rw])
        win_times_combined.append(win_times_rw[idx_rw])
        win_activities_combined.append(win_activities_rw[idx_rw])
    
    # Process unmatched left wrist windows (use single prediction)
    for idx_lw in unmatched_lw:
        probs_combined.append(probs_lw[idx_lw])
        y_true_combined.append(y_true_lw[idx_lw])
        win_times_combined.append(win_times_lw[idx_lw])
        win_activities_combined.append(win_activities_lw[idx_lw])
    
    if len(probs_combined) == 0:
        return None
    
    probs = np.array(probs_combined)
    y_true = np.array(y_true_combined)
    y_pred = (probs > CONF_THRESH).astype(int)
    discontinuity_times = get_discontinuity_times(times_rw)
    
    prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
    
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'accuracy': acc_score,
        'confusion_matrix': cm,
        'probs': probs,
        'y_true': y_true,
        'y_pred': y_pred,
        'win_times': np.array(win_times_combined),
        'win_activities': np.array(win_activities_combined),
        'discontinuity_times': discontinuity_times,
    }


# def evaluate_both_wrists_voting_AND(model, device, folder_path, folder_name):
#     """
#     Evaluate both wrists using majority voting (AND logic). if both wrists predict gait, then combined prediction is gait. Otherwise non-gait.
#     Run inference separately on each wrist, combine predictions via voting.
#     Fall back to single wrist if one is missing data.
#     """
#     rw_data = load_wrist_data(folder_path, 'right', folder_name)
#     lw_data = load_wrist_data(folder_path, 'left', folder_name)
    
#     if rw_data is None or lw_data is None:
#         return None
    
#     times_rw, acc_rw, y_binary_rw, activities_rw = rw_data
#     times_lw, acc_lw, y_binary_lw, activities_lw = lw_data
    
#     # Extract windows from both wrists (full data, no trimming)
#     wins_rw, y_true_rw, win_times_rw, win_activities_rw = extract_windows_with_gaps_and_activity(
#         times_rw, acc_rw, y_binary_rw, activities_rw
#     )
    
#     wins_lw, y_true_lw, win_times_lw, win_activities_lw = extract_windows_with_gaps_and_activity(
#         times_lw, acc_lw, y_binary_lw, activities_lw
#     )
    
#     if wins_rw is None and wins_lw is None:
#         return None
    
#     # If only one wrist has data, use that
#     if wins_rw is None:
#         probs = run_inference(model, wins_lw, device)
#         y_pred = (probs > CONF_THRESH).astype(int)
#         discontinuity_times = get_discontinuity_times(times_lw)
#         prec, rec, f1, acc_score, cm = compute_metrics(y_true_lw, y_pred)
#         return {
#             'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
#             'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_lw, 'y_pred': y_pred,
#             'win_times': win_times_lw, 'win_activities': win_activities_lw,
#             'discontinuity_times': discontinuity_times,
#         }
    
#     if wins_lw is None:
#         probs = run_inference(model, wins_rw, device)
#         y_pred = (probs > CONF_THRESH).astype(int)
#         discontinuity_times = get_discontinuity_times(times_rw)
#         prec, rec, f1, acc_score, cm = compute_metrics(y_true_rw, y_pred)
#         return {
#             'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
#             'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_rw, 'y_pred': y_pred,
#             'win_times': win_times_rw, 'win_activities': win_activities_rw,
#             'discontinuity_times': discontinuity_times,
#         }
    
#     # Match windows by time
#     matches, unmatched_rw, unmatched_lw = match_windows_by_time(win_times_rw, win_times_lw)
    
#     # Build combined results
#     probs_combined = []
#     y_pred_combined = []
#     y_true_combined = []
#     win_times_combined = []
#     win_activities_combined = []
    
#     # Run inference
#     probs_rw = run_inference(model, wins_rw, device)
#     probs_lw = run_inference(model, wins_lw, device)
#     y_pred_rw = (probs_rw > CONF_THRESH).astype(int)
#     y_pred_lw = (probs_lw > CONF_THRESH).astype(int)
    
#     # Process matched windows (voting: AND logic)
#     for idx_rw, idx_lw in matches:
#         # AND voting: both must agree on gait
#         vote = ((y_pred_rw[idx_rw] + y_pred_lw[idx_lw]) >= 2)
#         probs_combined.append(min(probs_rw[idx_rw], probs_lw[idx_lw]))  # Use minimum probability for conservative estimate
#         y_pred_combined.append(int(vote))
#         y_true_combined.append(y_true_rw[idx_rw])
#         win_times_combined.append(win_times_rw[idx_rw])
#         win_activities_combined.append(win_activities_rw[idx_rw])
    
#     # Process unmatched windows (use single wrist prediction)
#     for idx_rw in unmatched_rw:
#         probs_combined.append(probs_rw[idx_rw])
#         y_pred_combined.append(y_pred_rw[idx_rw])
#         y_true_combined.append(y_true_rw[idx_rw])
#         win_times_combined.append(win_times_rw[idx_rw])
#         win_activities_combined.append(win_activities_rw[idx_rw])
    
#     for idx_lw in unmatched_lw:
#         probs_combined.append(probs_lw[idx_lw])
#         y_pred_combined.append(y_pred_lw[idx_lw])
#         y_true_combined.append(y_true_lw[idx_lw])
#         win_times_combined.append(win_times_lw[idx_lw])
#         win_activities_combined.append(win_activities_lw[idx_lw])
    
#     if len(probs_combined) == 0:
#         return None
    
#     probs = np.array(probs_combined)
#     y_pred = np.array(y_pred_combined)
#     y_true = np.array(y_true_combined)
#     discontinuity_times = get_discontinuity_times(times_rw)
    
#     prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
    
#     return {
#         'precision': prec,
#         'recall': rec,
#         'f1': f1,
#         'accuracy': acc_score,
#         'confusion_matrix': cm,
#         'probs': probs,
#         'y_true': y_true,
#         'y_pred': y_pred,
#         'win_times': np.array(win_times_combined),
#         'win_activities': np.array(win_activities_combined),
#         'discontinuity_times': discontinuity_times,
#     }

# def evaluate_both_wrists_voting_OR(model, device, folder_path, folder_name):
#     """
#     Evaluate both wrists using majority voting (OR logic). if either wrist predicts gait, then combined prediction is gait. Otherwise non-gait.
#     Run inference separately on each wrist, combine predictions via voting.
#     Fall back to single wrist if one is missing data.
#     """
#     rw_data = load_wrist_data(folder_path, 'right', folder_name)
#     lw_data = load_wrist_data(folder_path, 'left', folder_name)
    
#     if rw_data is None or lw_data is None:
#         return None
    
#     times_rw, acc_rw, y_binary_rw, activities_rw = rw_data
#     times_lw, acc_lw, y_binary_lw, activities_lw = lw_data
    
#     # Extract windows from both wrists (full data, no trimming)
#     wins_rw, y_true_rw, win_times_rw, win_activities_rw = extract_windows_with_gaps_and_activity(
#         times_rw, acc_rw, y_binary_rw, activities_rw
#     )
    
#     wins_lw, y_true_lw, win_times_lw, win_activities_lw = extract_windows_with_gaps_and_activity(
#         times_lw, acc_lw, y_binary_lw, activities_lw
#     )
    
#     if wins_rw is None and wins_lw is None:
#         return None
    
#     # If only one wrist has data, use that
#     if wins_rw is None:
#         probs = run_inference(model, wins_lw, device)
#         y_pred = (probs > CONF_THRESH).astype(int)
#         discontinuity_times = get_discontinuity_times(times_lw)
#         prec, rec, f1, acc_score, cm = compute_metrics(y_true_lw, y_pred)
#         return {
#             'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
#             'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_lw, 'y_pred': y_pred,
#             'win_times': win_times_lw, 'win_activities': win_activities_lw,
#             'discontinuity_times': discontinuity_times,
#         }
    
#     if wins_lw is None:
#         probs = run_inference(model, wins_rw, device)
#         y_pred = (probs > CONF_THRESH).astype(int)
#         discontinuity_times = get_discontinuity_times(times_rw)
#         prec, rec, f1, acc_score, cm = compute_metrics(y_true_rw, y_pred)
#         return {
#             'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
#             'confusion_matrix': cm, 'probs': probs, 'y_true': y_true_rw, 'y_pred': y_pred,
#             'win_times': win_times_rw, 'win_activities': win_activities_rw,
#             'discontinuity_times': discontinuity_times,
#         }
    
#     # Match windows by time
#     matches, unmatched_rw, unmatched_lw = match_windows_by_time(win_times_rw, win_times_lw)
    
#     # Build combined results
#     probs_combined = []
#     y_pred_combined = []
#     y_true_combined = []
#     win_times_combined = []
#     win_activities_combined = []
    
#     # Run inference
#     probs_rw = run_inference(model, wins_rw, device)
#     probs_lw = run_inference(model, wins_lw, device)
#     y_pred_rw = (probs_rw > CONF_THRESH).astype(int)
#     y_pred_lw = (probs_lw > CONF_THRESH).astype(int)
    
#     # Process matched windows (voting: AND logic)
#     for idx_rw, idx_lw in matches:
#         # OR voting: either must agree on gait
#         vote = ((y_pred_rw[idx_rw] + y_pred_lw[idx_lw]) >= 1)
#         probs_combined.append(max(probs_rw[idx_rw], probs_lw[idx_lw]))  # Use maximum probability for optimistic estimate
#         y_pred_combined.append(int(vote))
#         y_true_combined.append(y_true_rw[idx_rw])
#         win_times_combined.append(win_times_rw[idx_rw])
#         win_activities_combined.append(win_activities_rw[idx_rw])
    
#     # Process unmatched windows (use single wrist prediction)
#     for idx_rw in unmatched_rw:
#         probs_combined.append(probs_rw[idx_rw])
#         y_pred_combined.append(y_pred_rw[idx_rw])
#         y_true_combined.append(y_true_rw[idx_rw])
#         win_times_combined.append(win_times_rw[idx_rw])
#         win_activities_combined.append(win_activities_rw[idx_rw])
    
#     for idx_lw in unmatched_lw:
#         probs_combined.append(probs_lw[idx_lw])
#         y_pred_combined.append(y_pred_lw[idx_lw])
#         y_true_combined.append(y_true_lw[idx_lw])
#         win_times_combined.append(win_times_lw[idx_lw])
#         win_activities_combined.append(win_activities_lw[idx_lw])
    
#     if len(probs_combined) == 0:
#         return None
    
#     probs = np.array(probs_combined)
#     y_pred = np.array(y_pred_combined)
#     y_true = np.array(y_true_combined)
#     discontinuity_times = get_discontinuity_times(times_rw)
    
#     prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
    
#     return {
#         'precision': prec,
#         'recall': rec,
#         'f1': f1,
#         'accuracy': acc_score,
#         'confusion_matrix': cm,
#         'probs': probs,
#         'y_true': y_true,
#         'y_pred': y_pred,
#         'win_times': np.array(win_times_combined),
#         'win_activities': np.array(win_activities_combined),
#         'discontinuity_times': discontinuity_times,
#     }

def evaluate_qsense_clinic_wrist_comparison(model, device, dataset_path):
    """
    Evaluate QSense Clinic dataset with three conditions:
    1. Affected wrist only
    2. Unaffected wrist only
    3. Both wrists (combined)
    """
    print("\n" + "=" * 70)
    print("EVALUATING: QSense Clinic - Wrist Comparison Analysis")
    print("=" * 70)
    
    results = []
    
    if not os.path.isdir(dataset_path):
        print(f"Path not found: {dataset_path}")
        return results
    
    # Evaluate each subject
    for subject_idx, folder in enumerate(sorted(os.listdir(dataset_path))):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        folder_lower = folder.lower()
        if not folder_lower.startswith('sub'):
            print(f"\nSkipping non-subject folder: {folder}")
            continue

        subject_digits = ''.join(ch for ch in folder if ch.isdigit())
        if subject_digits == '':
            print(f"\nSkipping folder with unrecognized subject id: {folder}")
            continue

        subject_num = int(subject_digits)

        has_affected_mapping = 1 <= subject_num <= len(affected_wrist_patient)
        if has_affected_mapping:
            affected_wrist = affected_wrist_patient[subject_num - 1]
            unaffected_wrist = 'LW' if affected_wrist == 'RW' else 'RW'
            print(f"\n--- {folder} (Affected: {affected_wrist}, Unaffected: {unaffected_wrist}) ---")

            first_wrist_name = 'right' if affected_wrist == 'RW' else 'left'
            second_wrist_name = 'right' if unaffected_wrist == 'RW' else 'left'
            first_label = 'affected'
            second_label = 'unaffected'
            first_condition = 'affected'
            second_condition = 'unaffected'
            affected_wrist_value = affected_wrist
        else:
            print(f"\n--- {folder} (Affected side unknown; evaluating right/left wrists) ---")
            first_wrist_name = 'right'
            second_wrist_name = 'left'
            first_label = 'right'
            second_label = 'left'
            first_condition = 'right'
            second_condition = 'left'
            affected_wrist_value = 'Unknown'

        # 1. Evaluate first wrist (affected if known, else right)
        print(f"  Evaluating {first_wrist_name.upper()} wrist ({first_label})...")
        first_result = evaluate_specific_wrist(model, device, folder_path, folder, first_wrist_name)
        
        if first_result:
            print(f"    Precision: {first_result['precision']:.3f} | "
                  f"Recall: {first_result['recall']:.3f} | "
                  f"F1: {first_result['f1']:.3f} | "
                  f"Accuracy: {first_result['accuracy']:.3f}")
            results.append({
                'subject': folder,
                'subject_num': subject_num,
                'condition': first_condition,
                'condition_name': first_label.capitalize(),
                'affected_wrist': affected_wrist_value,
                'wrist_name': first_wrist_name,
                **first_result
            })
        else:
            print(f"    Could not evaluate {first_wrist_name} wrist")
        
        # 2. Evaluate second wrist (unaffected if known, else left)
        print(f"  Evaluating {second_wrist_name.upper()} wrist ({second_label})...")
        second_result = evaluate_specific_wrist(model, device, folder_path, folder, second_wrist_name)
        
        if second_result:
            print(f"    Precision: {second_result['precision']:.3f} | "
                  f"Recall: {second_result['recall']:.3f} | "
                  f"F1: {second_result['f1']:.3f} | "
                  f"Accuracy: {second_result['accuracy']:.3f}")
            results.append({
                'subject': folder,
                'subject_num': subject_num,
                'condition': second_condition,
                'condition_name': second_label.capitalize(),
                'affected_wrist': affected_wrist_value,
                'wrist_name': second_wrist_name,
                **second_result
            })
        else:
            print(f"    Could not evaluate {second_wrist_name} wrist")
        
        # 3. Evaluate both wrists with different fusion strategies
        fusion_strategies = {
            'averaged_acc': ('ACC averaged', evaluate_both_wrists),
            'prob_average': ('Probability averaged', evaluate_both_wrists_prob_average),
            'prob_max': ('Probability max', evaluate_both_wrists_prob_max),
            'prob_min': ('Probability min', evaluate_both_wrists_prob_min),
            # 'voting_AND': ('Voting (AND)', evaluate_both_wrists_voting_AND),
            # 'voting_OR': ('Voting (OR)', evaluate_both_wrists_voting_OR),
        }
        
        for strategy_key, (strategy_name, strategy_func) in fusion_strategies.items():
            print(f"  Evaluating BOTH wrists ({strategy_name})...")
            result = strategy_func(model, device, folder_path, folder)
            
            if result:
                print(f"    Precision: {result['precision']:.3f} | "
                      f"Recall: {result['recall']:.3f} | "
                      f"F1: {result['f1']:.3f} | "
                      f"Accuracy: {result['accuracy']:.3f}")
                results.append({
                    'subject': folder,
                    'subject_num': subject_num,
                    'condition': strategy_key,
                    'condition_name': strategy_name,
                    'affected_wrist': affected_wrist_value,
                    'wrist_name': 'both',
                    **result
                })
            else:
                print(f"    Could not evaluate both wrists ({strategy_name})")
    
    return results


def create_comparison_dataframe(results):
    """Create a pandas DataFrame from results for easy analysis."""
    if not results:
        return None
    
    df_data = []
    for r in results:
        condition_name = r.get('condition_name', r['condition'])
        df_data.append({
            'Subject': r['subject'],
            'Subject_Num': r['subject_num'],
            'Condition': r['condition'],
            'Condition_Name': condition_name,
            'Affected_Wrist': r['affected_wrist'],
            'Wrist_Name': r['wrist_name'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1': r['f1'],
            'Accuracy': r['accuracy'],
        })
    
    return pd.DataFrame(df_data)


def create_comparison_plots(results, plots_dir):
    """Create comprehensive comparison plots for all fusion strategies."""
    os.makedirs(plots_dir, exist_ok=True)
    
    df = create_comparison_dataframe(results)
    if df is None or len(df) == 0:
        print("No results to plot")
        return
    
    subjects = sorted(df['Subject'].unique())
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
    
    # Separate single-wrist and fusion conditions
    single_conditions = ['affected', 'unaffected', 'right', 'left']
    fusion_conditions = ['averaged_acc', 'prob_average', 'prob_max', 'prob_min', 'voting_AND', 'voting_OR']
    
    # Plot 1: Detailed comparison with all conditions for each subject
    fig, axes = plt.subplots(len(subjects), len(metrics), figsize=(18, 4 * len(subjects)))
    if len(subjects) == 1:
        axes = axes.reshape(1, -1)
    
    # Color map for conditions
    color_map = {
        'affected': '#FF6B6B',
        'unaffected': '#4ECDC4',
        'right': '#1F77B4',
        'left': '#2CA02C',
        'averaged_acc': "#000000",
        'prob_average': "#1900FC",
        'prob_max': "#E6BB2C",
        'prob_min': '#DFE6E9',
        # 'voting_AND': '#A29BFE',
        # 'voting_OR': "#FF9D00",
    }
    
    for sub_idx, subject in enumerate(subjects):
        sub_data = df[df['Subject'] == subject].sort_values('Condition')
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[sub_idx, metric_idx] if len(subjects) > 1 else axes[metric_idx]
            
            conditions = sub_data['Condition'].values
            condition_names = sub_data['Condition_Name'].values
            values = sub_data[metric].values
            colors = [color_map.get(c, '#999999') for c in conditions]
            
            bars = ax.bar(range(len(conditions)), values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylim([0, 1])
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f"{subject} - {metric}", fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(condition_names, rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'wrist_fusion_detailed.png'), dpi=300, bbox_inches='tight')
    print(f"Saved detailed comparison plot to {plots_dir}/wrist_fusion_detailed.png")
    plt.close()
    
    # Plot 2: Fusion strategies comparison (averaged across conditions)
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Get all conditions and their means across subjects
        for condition in single_conditions + fusion_conditions:
            cond_data = df[df['Condition'] == condition]
            if len(cond_data) > 0:
                condition_name = cond_data['Condition_Name'].iloc[0] if 'Condition_Name' in cond_data.columns else condition
                values = cond_data.sort_values('Subject')[metric].values
                subject_labels = cond_data.sort_values('Subject')['Subject'].values
                
                color = color_map.get(condition, '#999999')
                ax.plot(subject_labels, values, marker='o', label=condition_name.replace(' (', '\n('), 
                       linewidth=2.5, markersize=8, color=color, alpha=0.8)
        
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xlabel('Subject', fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'wrist_fusion_trends.png'), dpi=300, bbox_inches='tight')
    print(f"Saved trend comparison plot to {plots_dir}/wrist_fusion_trends.png")
    plt.close()


def create_method_roc_pr_plots(results, plots_dir):
    """Create ROC and PR plots grouped by method across all subjects."""
    os.makedirs(plots_dir, exist_ok=True)

    method_groups = defaultdict(lambda: {'y_true': [], 'probs': []})
    for r in results:
        if 'y_true' not in r or 'probs' not in r:
            continue
        method_name = r.get('condition_name', r.get('condition', 'unknown'))
        method_groups[method_name]['y_true'].append(np.asarray(r['y_true']).astype(int))
        method_groups[method_name]['probs'].append(np.asarray(r['probs']).astype(float))

    if len(method_groups) == 0:
        print('No results available for ROC/PR plotting.')
        return

    roc_dir = os.path.join(plots_dir, 'ROC')
    pr_dir = os.path.join(plots_dir, 'PR')
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    threshold_rows = []
    threshold_by_method = {}

    # Combined ROC plot with one curve per method
    plt.figure(figsize=(10, 8))
    any_roc = False

    for method, vals in method_groups.items():
        y_true = np.concatenate(vals['y_true']) if vals['y_true'] else np.array([])
        probs = np.concatenate(vals['probs']) if vals['probs'] else np.array([])

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            print(f"Skipping ROC for {method}: both classes are required.")
            continue

        fpr, tpr, thresholds = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        any_roc = True
        threshold_info = find_optimal_thresholds(y_true, probs, beta=1.0)

        youden_threshold = np.nan
        youden_j = np.nan
        youden_fpr = np.nan
        youden_tpr = np.nan
        f1_threshold = np.nan
        f1_score = np.nan
        f1_precision = np.nan
        f1_recall = np.nan

        if threshold_info is not None:
            if 'youden_j' in threshold_info:
                youden_threshold = threshold_info['youden_j']['threshold']
                youden_j = threshold_info['youden_j']['j_score']
                youden_fpr = threshold_info['youden_j']['fpr']
                youden_tpr = threshold_info['youden_j']['tpr']
            if 'f_beta' in threshold_info:
                f1_threshold = threshold_info['f_beta']['threshold']
                f1_score = threshold_info['f_beta']['f_beta']
                f1_precision = threshold_info['f_beta']['precision']
                f1_recall = threshold_info['f_beta']['recall']

        threshold_rows.append({
            'method': method,
            'roc_auc': float(roc_auc),
            'average_precision': np.nan,
            'youden_threshold': youden_threshold,
            'youden_j': youden_j,
            'youden_fpr': youden_fpr,
            'youden_tpr': youden_tpr,
            'f1_threshold': f1_threshold,
            'f1_score': f1_score,
            'f1_precision': f1_precision,
            'f1_recall': f1_recall,
        })
        threshold_by_method[method] = {
            'youden_threshold': youden_threshold,
            'youden_fpr': youden_fpr,
            'youden_tpr': youden_tpr,
            'f1_threshold': f1_threshold,
            'f1_precision': f1_precision,
            'f1_recall': f1_recall,
        }

        safe_name = method.replace(' ', '_').replace('(', '').replace(')', '').lower()
        roc_points = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})
        roc_points.to_csv(os.path.join(roc_dir, f'roc_points_{safe_name}.csv'), index=False)

        # Save per-method ROC plot
        plt_method = plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {method}')
        if np.isfinite(youden_fpr) and np.isfinite(youden_tpr):
            plt.scatter(
                [youden_fpr],
                [youden_tpr],
                s=50,
                color='red',
                zorder=3,
                label=f"Youden t={youden_threshold:.3f}",
            )
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(roc_dir, f'roc_{safe_name}.png'), dpi=220, bbox_inches='tight')
        plt.close(plt_method)

        plt.figure(1)
        plt.plot(fpr, tpr, linewidth=2, label=f"{method} (AUC={roc_auc:.3f})")

    if any_roc:
        plt.figure(1)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Method')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        combined_roc_path = os.path.join(roc_dir, 'roc_methods_combined.png')
        plt.savefig(combined_roc_path, dpi=220, bbox_inches='tight')
        print(f"Saved combined ROC plot to {combined_roc_path}")
    else:
        print('No ROC curves generated (insufficient class variation).')
    plt.close('all')

    # Combined PR plot with one curve per method
    plt.figure(figsize=(10, 8))
    any_pr = False

    for method, vals in method_groups.items():
        y_true = np.concatenate(vals['y_true']) if vals['y_true'] else np.array([])
        probs = np.concatenate(vals['probs']) if vals['probs'] else np.array([])

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            print(f"Skipping PR for {method}: both classes are required.")
            continue

        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        any_pr = True

        if threshold_rows:
            threshold_rows[-1]['average_precision'] = float(ap)

        safe_name = method.replace(' ', '_').replace('(', '').replace(')', '').lower()
        pr_points = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'threshold': np.r_[thresholds, np.nan],
        })
        pr_points.to_csv(os.path.join(pr_dir, f'pr_points_{safe_name}.csv'), index=False)

        # Save per-method PR plot
        plt_method = plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, linewidth=2, label=f"AP={ap:.3f}")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR - {method}')
        method_thr = threshold_by_method.get(method, {})
        method_f1_recall = method_thr.get('f1_recall', np.nan)
        method_f1_precision = method_thr.get('f1_precision', np.nan)
        method_f1_threshold = method_thr.get('f1_threshold', np.nan)
        if np.isfinite(method_f1_recall) and np.isfinite(method_f1_precision):
            plt.scatter(
                [method_f1_recall],
                [method_f1_precision],
                s=50,
                color='red',
                zorder=3,
                label=f"Best F1 t={method_f1_threshold:.3f}",
            )
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pr_dir, f'pr_{safe_name}.png'), dpi=220, bbox_inches='tight')
        plt.close(plt_method)

        plt.figure(1)
        plt.plot(recall, precision, linewidth=2, label=f"{method} (AP={ap:.3f})")

    if any_pr:
        plt.figure(1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves by Method')
        plt.legend(loc='lower left', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        combined_pr_path = os.path.join(pr_dir, 'pr_methods_combined.png')
        plt.savefig(combined_pr_path, dpi=220, bbox_inches='tight')
        print(f"Saved combined PR plot to {combined_pr_path}")
    else:
        print('No PR curves generated (insufficient class variation).')
    plt.close('all')

    if threshold_rows:
        thresholds_df = pd.DataFrame(threshold_rows)
        thresholds_csv = os.path.join(plots_dir, 'optimal_thresholds_by_method.csv')
        thresholds_df.to_csv(thresholds_csv, index=False)
        print(f"Saved threshold summary to {thresholds_csv}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = load_finetuned_model(WEIGHTS_PATH).to(device)

    # Evaluate QSense Clinic dataset with wrist comparison
    results = evaluate_qsense_clinic_wrist_comparison(model, device, QSENSE_CLINIC)
    
    # Create comparison dataframe
    df = create_comparison_dataframe(results)
    
    if df is not None and len(df) > 0:
        print("\n" + "=" * 70)
        print("SUMMARY RESULTS - ALL FUSION STRATEGIES")
        print("=" * 70)
        print(df.to_string(index=False))
        
        # Save to CSV
        results_csv = os.path.join(RESULTS_DIR, 'StrokeNet_clinic_wrist_comparison_all_methods.csv')
        df.to_csv(results_csv, index=False)
        print(f"\nDetailed results saved to {results_csv}")
        
        # Create plots
        plots_subdir = os.path.join(PLOTS_DIR, 'QSense_data_clinic', 'wrist_comparison', 'StrokeNet')
        create_comparison_plots(results, plots_subdir)
        create_method_roc_pr_plots(results, plots_subdir)
        
        # Print subject-wise comparison
        print("\n" + "=" * 70)
        print("SUBJECT-WISE COMPARISON")
        print("=" * 70)
        for subject in df['Subject'].unique():
            sub_df = df[df['Subject'] == subject]
            print(f"\n{subject}:")
            print(sub_df[['Condition', 'Precision', 'Recall', 'F1', 'Accuracy']].to_string(index=False))


if __name__ == "__main__":
    main()
