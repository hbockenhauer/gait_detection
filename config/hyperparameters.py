"""Centralized hyperparameter presets for training and evaluation."""

ELDERNET = {
    'window_size': 300,
    'step_size': 30,
    'sample_rate_hz': 30,
}

STROKENET = {
    'window_size': 100,
    'step_size': 50,
    'sample_rate_hz': 50,
    'confidence_threshold': 0.5,
}

QSENSE = {
    'sample_rate_hz': 50,
    'gap_threshold_sec': 0.1,
}
