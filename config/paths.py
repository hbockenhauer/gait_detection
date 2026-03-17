"""
Centralized path management for the gait_detection project.
All dataset, model, and output paths are defined here.
Use this module instead of manually calculating paths in scripts.
"""

import os


def get_project_root():
    """
    Auto-detect project root directory from any location in the repo.
    Works from any subfolder (models/, config/, analysis/, etc.)
    Returns absolute path to gait_detection directory.
    """
    current = os.path.abspath(__file__)
    
    # Walk up directory tree looking for gait_detection folder
    while current != os.path.dirname(current):
        if os.path.basename(current) == 'gait_detection':
            return current
        current = os.path.dirname(current)
    
    raise RuntimeError("Could not find 'gait_detection' project root")


# Core paths
PROJECT_ROOT = get_project_root()
DATASETS_ROOT = os.path.join(PROJECT_ROOT, 'Datasets')
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_ROOT, 'plots')
RESULTS_DIR = os.path.join(OUTPUTS_ROOT, 'results')
LOGS_DIR = os.path.join(OUTPUTS_ROOT, 'logs')

# Model paths
ELDERNET_DIR = os.path.join(MODELS_ROOT, 'ElderNet')
STROKENET_DIR = os.path.join(MODELS_ROOT, 'StrokeNet')
STRACZKIEWICZ_DIR = os.path.join(MODELS_ROOT, 'mstraczkiewicz')

# Weights
ELDERNET_WEIGHTS = os.path.join(ELDERNET_DIR, 'eldernet_finetuned.pth')
STROKENET_WEIGHTS = os.path.join(STROKENET_DIR, 'eldernet_finetuned.pth')

# Dataset paths
HMP_PATH = os.path.join(DATASETS_ROOT, 'HMP_Dataset')
QSENSE_DATA = os.path.join(DATASETS_ROOT, 'QSense_data')
QSENSE_EDGE = os.path.join(DATASETS_ROOT, 'QSense_data_edge')
QSENSE_MIXED = os.path.join(DATASETS_ROOT, 'QSense_data_mixed')
QSENSE_CLINIC = os.path.join(DATASETS_ROOT, 'QSense_data_clinic')
WEARGAIT_PD = os.path.join(DATASETS_ROOT, 'WearGait', 'WearGait-PD')
WEARGAIT_CTRL = os.path.join(DATASETS_ROOT, 'WearGait', 'WearGait-Ctrl')
WISDM_PATH = os.path.join(DATASETS_ROOT, 'wisdm-dataset', 'raw', 'watch', 'accel')
FREELIVING_PATH = os.path.join(DATASETS_ROOT, 'Free_living')
BIOCLITE_PATH = os.path.join(DATASETS_ROOT, 'Bioclite', 'data_6activities_plain.mat')

def get_plot_dir(dataset: str, model: str) -> str:
    """Return the canonical output path for plots: outputs/plots/{dataset}/{model}/"""
    return os.path.join(PLOTS_DIR, dataset, model)


# Convenience constants — outputs/plots/{dataset}/{model}/
ELDERNET_PLOTS      = get_plot_dir('HMP',             'eldernet')
ELDERNET_DUAL_PLOTS = get_plot_dir('WearGait',        'eldernet')
ELDERNET_WISDM_PLOTS = get_plot_dir('WISDM',          'eldernet')
STROKENET_PLOTS     = get_plot_dir('QSense_data',     'strokenet')


def validate_dataset_paths():
    """
    Check if all expected dataset directories exist.
    Useful for debugging setup issues.
    Returns a dict of path -> exists (True/False)
    """
    paths_to_check = {
        'HMP_PATH': HMP_PATH,
        'QSENSE_DATA': QSENSE_DATA,
        'QSENSE_EDGE': QSENSE_EDGE,
        'QSENSE_MIXED': QSENSE_MIXED,
        'WEARGAIT_PD': WEARGAIT_PD,
        'WEARGAIT_CTRL': WEARGAIT_CTRL,
        'WISDM_PATH': WISDM_PATH,
        'FREELIVING_PATH': FREELIVING_PATH,
    }
    
    return {name: os.path.exists(path) for name, path in paths_to_check.items()}


if __name__ == '__main__':
    # Quick validation script
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Datasets Root: {DATASETS_ROOT}")
    print(f"Models Root: {MODELS_ROOT}")
    print(f"Outputs Root: {OUTPUTS_ROOT}")
    print("\nDataset Availability:")
    for name, exists in validate_dataset_paths().items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
