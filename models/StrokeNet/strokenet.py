"""Thin entrypoint for StrokeNet evaluation.

All dataset evaluators and combined ROC/PR logic live in
`strokenet_other_datasets.py` to avoid duplicated implementations.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import psutil
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

    # Save per-subject results
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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_subject_csv = os.path.join(RESULTS_DIR, 'strokenet_cross_dataset_per_subject.csv')
    global_csv      = os.path.join(RESULTS_DIR, 'strokenet_cross_dataset_global.csv')
    pd.DataFrame(all_rows).to_csv(per_subject_csv, index=False)
    pd.DataFrame(global_rows).to_csv(global_csv, index=False)
    print(f"\nSaved per-subject results : {per_subject_csv}")
    print(f"Saved global summary      : {global_csv}")


if __name__ == '__main__':
    main()