"""Thin entrypoint for StrokeNet evaluation.

All dataset evaluators and plotting functions are implemented in `strokenet_utils.py` to keep this file 
clean and focused on the overall evaluation flow. This also allows us to easily reuse the same 
evaluation code for other datasets in the future by simply calling the relevant functions from
 `strokenet_utils.py` without needing to modify this main script.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.StrokeNet.strokenet_utils import (
    load_finetuned_model,
    evaluate_wisdm,
    evaluate_weargait,
    evaluate_hmp,
    evaluate_bioclite,
    evaluate_qsense_dataset,
    evaluate_free_living,
    plot_subject_timeline,
    WEIGHTS_PATH,
    QSENSE_PATHS,
    FREE_LIVING_PATH,    FREE_LIVING_DATASET_NAME,
)
from config.paths import (
    PLOTS_DIR as OUTPUT_PLOTS_DIR,
    RESULTS_DIR,
)
import utils.plot_ROC_PR as plot_ROC_PR

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = load_finetuned_model(WEIGHTS_PATH).to(device)

    wisdm_results,    wisdm_global    = evaluate_wisdm(model, device)
    weargait_results, weargait_global = evaluate_weargait(model, device)
    hmp_results, hmp_global = evaluate_hmp(model, device)
    bioclite_results, bioclite_global = evaluate_bioclite(model, device)

    qsense_results_all = []
    qsense_globals = []
    for qsense_path in QSENSE_PATHS:
        qs_results, qs_global = evaluate_qsense_dataset(model, device, qsense_path)
        qsense_results_all.extend(qs_results)
        qsense_globals.append({'dataset': os.path.basename(os.path.normpath(qsense_path)), **qs_global})

    free_results, free_global = evaluate_free_living(model, device, FREE_LIVING_PATH)

    all_results = (
        wisdm_results
        + weargait_results
        + hmp_results
        + bioclite_results
        + qsense_results_all
        + free_results
    )
    plot_subject_timeline(all_results, OUTPUT_PLOTS_DIR)
    plot_ROC_PR.plot_roc_curves(all_results, OUTPUT_PLOTS_DIR)
    plot_ROC_PR.plot_pr_curves(all_results, OUTPUT_PLOTS_DIR)

    # Save per-dataset results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # WISDM
    if wisdm_results:
        wisdm_rows = [{
            'subject': r['subject'], 'dataset': 'WISDM',
            'precision': r['precision'], 'recall': r['recall'],
            'f1': r['f1'], 'accuracy': r['accuracy']
        } for r in wisdm_results]
        wisdm_csv = os.path.join(RESULTS_DIR, 'strokenet_WISDM_subject_metrics.csv')
        pd.DataFrame(wisdm_rows).to_csv(wisdm_csv, index=False)
        print(f"Saved WISDM results: {wisdm_csv}")
    
    # WearGait
    if weargait_results:
        weargait_rows = [{
            'subject': r['subject'], 'wrist': r.get('wrist', 'N/A'), 'dataset': 'WearGait',
            'precision': r['precision'], 'recall': r['recall'],
            'f1': r['f1'], 'accuracy': r['accuracy']
        } for r in weargait_results]
        weargait_csv = os.path.join(RESULTS_DIR, 'strokenet_WearGait_subject_metrics.csv')
        pd.DataFrame(weargait_rows).to_csv(weargait_csv, index=False)
        print(f"Saved WearGait results: {weargait_csv}")
    
    # HMP
    if hmp_results:
        hmp_rows = [{
            'subject': r['subject'], 'dataset': 'HMP',
            'precision': r['precision'], 'recall': r['recall'],
            'f1': r['f1'], 'accuracy': r['accuracy']
        } for r in hmp_results]
        hmp_csv = os.path.join(RESULTS_DIR, 'strokenet_HMP_subject_metrics.csv')
        pd.DataFrame(hmp_rows).to_csv(hmp_csv, index=False)
        print(f"Saved HMP results: {hmp_csv}")
    
    # BIOCLITE
    if bioclite_results:
        bioclite_rows = [{
            'subject': r['subject'], 'dataset': 'BIOCLITE',
            'precision': r['precision'], 'recall': r['recall'],
            'f1': r['f1'], 'accuracy': r['accuracy']
        } for r in bioclite_results]
        bioclite_csv = os.path.join(RESULTS_DIR, 'strokenet_BIOCLITE_subject_metrics.csv')
        pd.DataFrame(bioclite_rows).to_csv(bioclite_csv, index=False)
        print(f"Saved BIOCLITE results: {bioclite_csv}")

    # Aggregated results for all datasets
    all_rows = []
    for r in all_results:
        all_rows.append({
            'dataset':   r['dataset'],
            'subject':   r['subject'],
            'wrist':     r.get('wrist', 'N/A'),
            'precision': r['precision'],
            'recall':    r['recall'],
            'f1':        r['f1'],
            'accuracy':  r['accuracy']
        })

    # Save global summary
    global_rows = [
        {'dataset': 'WISDM',    **wisdm_global},
        {'dataset': 'WearGait', **weargait_global},
        {'dataset': 'HMP',      **hmp_global},
        {'dataset': 'BIOCLITE', **bioclite_global},
        *qsense_globals,
        {'dataset': FREE_LIVING_DATASET_NAME, **free_global},
    ]

    per_subject_csv = os.path.join(RESULTS_DIR, 'strokenet_cross_dataset_per_subject.csv')
    global_csv      = os.path.join(RESULTS_DIR, 'strokenet_cross_dataset_global.csv')
    pd.DataFrame(all_rows).to_csv(per_subject_csv, index=False)
    pd.DataFrame(global_rows).to_csv(global_csv, index=False)
    print(f"\nSaved aggregated per-subject results: {per_subject_csv}")
    print(f"Saved aggregated global summary: {global_csv}")


if __name__ == '__main__':
    main()