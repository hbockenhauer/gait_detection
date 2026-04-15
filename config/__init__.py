"""Configuration module for gait_detection project."""

from .paths import (
    PROJECT_ROOT,
    DATASETS_ROOT,
    MODELS_ROOT,
    OUTPUTS_ROOT,
    HMP_PATH,
    QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    WEARGAIT_PD,
    WEARGAIT_CTRL,
    WISDM_PATH,
    FREELIVING_PATH,
    BIOCLITE_PATH,
    ELDERNET_WEIGHTS,
    STROKENET_WEIGHTS,
    validate_dataset_paths,
)

__all__ = [
    'PROJECT_ROOT',
    'DATASETS_ROOT',
    'MODELS_ROOT',
    'OUTPUTS_ROOT',
    'HMP_PATH',
    'QSENSE_DATA',
    'QSENSE_EDGE',
    'QSENSE_MIXED',
    'WEARGAIT_PD',
    'WEARGAIT_CTRL',
    'WISDM_PATH',
    'FREELIVING_PATH',
    'BIOCLITE_PATH',
    'ELDERNET_WEIGHTS',
    'STROKENET_WEIGHTS',
    'validate_dataset_paths',
]
