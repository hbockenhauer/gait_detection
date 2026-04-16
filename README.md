# Gait Detection

Gait detection algorithm using wrist-worn Inertial Measurement Unit (IMU) data for patients with stroke.

## Project Overview

This project implements Machine Learning models (ElderNet/StrokeNet) and Signal Processing algorithms (mstraczkiewicz, Hickey, Kheirkhahan) for gait detection from wearable IMU sensors. It also includes real-time simulation pipelines for per-wrist and fused two-wrist gait detection. The models are evaluated across multiple public datasets (WISDM, HMP, WearGait, Bioclite) as well as stroke patient cohorts and self-recorded data using QSense Motion Capture IMUs.

## Directory Structure

```
gait_detection/
├── Algo_flowcharts.drawio     # Pipeline and algorithm flowcharts
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── annotation_tools/          # Data annotation and validation tools
│   ├── mixed_annotater.py     # Annotation tool for Multiple_Activities files
│   └── video_annotater.py     # Video frame labeling tool for Free_living data
│
├── config/                    # Configuration management and centralized paths
│   ├── hyperparameters.py     # Model hyperparameter configs
│   └── paths.py               # Dataset and model path definitions
│
├── Datasets/                  # All training and evaluation data
│   ├── Baseline/              # Self-recorded baseline activities (50Hz)
│   ├── Bioclite/              # 6-activity reference dataset (50Hz)
│   ├── Clinical/              # Data recorded on stroke patients (50Hz)
│   ├── Edge_Cases/            # Self-recorded edge cases (50Hz)
│   ├── Free_living/           # Stroke patient data (50Hz)
│   ├── HMP_Dataset/           # Fall detection dataset (32Hz)
│   ├── Multiple_Activities/   # Self-recorded mixed activities (50Hz)
│   ├── Qsense_tests/          # Internal test recordings and checks
│   ├── WearGait/              # Parkinson's and control cohorts (100Hz)
│   └── wisdm-dataset/         # Public activity recognition dataset (20Hz)
│
├── models/                    # Model implementations
│   ├── ElderNet/              # Gait detection model inference source code
│   ├── Hickey/                # Wrist-adapted Hickey gait sequence detection
│   ├── Kheirkhahan/           # Wrist and fused Kheirkhahan gait sequence detection
│   ├── mstraczkiewicz/        # Signal processing Straczkiewicz MATLAB implementations
│   ├── realtime/              # Real-time simulation and fused/per-wrist evaluation
│   └── StrokeNet/             # Fine-tuned variant for stroke patients

├── outputs/                   # Generated outputs
│   ├── logs/                  # Training and evaluation logs
│   ├── plots/                 # Visualizations and result plots
│   └── results/               # CSV results and metrics

└── utils/                     # Shared utilities and helpers
    ├── __init__.py            # Utility package initializer
    ├── comp_load.py           # Compute computational load of models
    ├── data_loaders.py        # Common dataset loading utilities
    ├── faulty_data.py         # Detection and handling helpers for faulty segments
    ├── hub_utils.py           # Shared helper functions for model pipelines
    ├── leg_comp.py            # Leg-side and condition comparison utilities
    ├── plot_accelerations.py  # Plot raw acceleration channels
    ├── plot_powerspec.py      # Plot power spectra of acceleration signals
    ├── plot_qsense_activities.py # Plot QSense activities and signal features
    ├── plot_ROC_PR.py         # Plot ROC and precision-recall curves
    ├── time_check.py          # Timing and runtime validation helpers
    └── visualization.py       # Generic plotting and visualization functions

```

## Getting Started

### Installation

1. Clone the repository:
	```bash
	git clone <repo-url>
	cd gait_detection
	```

2. Create a Python virtual environment:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	```

3. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

### Quick Start

#### Using Centralized Paths
Instead of hardcoding paths, use the centralized path configuration:
```python
from config import HMP_PATH, WEARGAIT_PD, OUTPUTS_ROOT, validate_dataset_paths

# Validate dataset setup
status = validate_dataset_paths()
```

## Models

### ElderNet
Core gait detection model trained on multiple public datasets. Supports inference on provided datasets.

Files:
- `models/ElderNet/eldernet_*.py` - Dataset-specific evaluation scripts
- `models/ElderNet/eldernet_comp_load.py` -  Script to determine computational load of ElderNet


### StrokeNet
Fine-tuned variant of ElderNet optimized for stroke patients. Used for activity-level metric computation.

Files:
- `models/StrokeNet/Energy_analysis.ipynb` - Jupyter Notebook for clinician to run end-of-day analysis
- `models/StrokeNet/energy_report.py` - Script to perform end-of-day energy analysis of patient
- `models/StrokeNet/helper_functions.py` - Necessary functions for Notebook 
- `models/StrokeNet/retrain_eldernet.py` - Fine-tuning script
- `models/StrokeNet/strokenet_comp_load.py` - Script to determine computational load of StrokeNet
- `models/StrokeNet/strokenet_upperarm.py` - Script to perform gait detection using sensor on upper arm for Clinical dataset
- `models/StrokeNet/strokenet_utils.py` - Script to run necessary data loaders and plotters for inference
- `models/StrokeNet/StrokeNet_weights.pth` - New model weights obtained from retrain_eldernet.py
- `models/StrokeNet/strokenet_wrist.py` - Script to analyse different wrist fusion strategies 
- `models/StrokeNet/strokenet.py` - Main inference pipeline for global cross-dataset evaluation
- `models/StrokeNet/__init__.py` - Package initializer

### Straczkiewicz

Reference MATLAB implementations of the Straczkiewicz gait detection pipeline, used for signal-processing-based walking detection, dataset benchmarking, and real-time simulation.

Files:
- `models/mstraczkiewicz/find_walking.m` - Core walking detector based on vector magnitude, CWT peaks, and cadence continuity checks.
- `models/mstraczkiewicz/find_continuous_dominant_peaks.m` - Helper that enforces continuity of dominant spectral peaks across windows.
- `models/mstraczkiewicz/load_weargait_data.m` - Loader for WearGait CSV files that separates right and left wrist streams.
- `models/mstraczkiewicz/MStra_QSense.m` - Batch evaluation on self-recorded data, Clinical using classic Straczkiewicz detector.
- `models/mstraczkiewicz/MStra_WearGait.m` - WearGait-PD evaluation using the classic Straczkiewicz detector.
- `models/mstraczkiewicz/Mstra_RT.m` - Adapted real-time simulation on free-living stroke patient data with timestamp gap handling.
- `models/mstraczkiewicz/MStra_RT_freeliving.m` - Adapted real-time evaluator for annotated Free_living CSV files.
- `models/mstraczkiewicz/MStra_RT_cross_dataset.m` - Cross-dataset real-time evaluation and summary generation using adapted algorithm
- `models/mstraczkiewicz/Mstra_RT_wrist_comparison.m` - Compares fusion strategies for Clinical data using adapted algorithm.
- `models/mstraczkiewicz/param_opt_QSense.m` - Bayesian optimization of detector thresholds on Multiple_Activities  data.
- `models/mstraczkiewicz/param_opt_freeliving.m` - Bayesian optimization of detector thresholds on Free_living data.


### Hickey
Wrist-adapted implementation of the Hickey gait sequence detection approach, including robust dataset processing for existing datasets and QSense cohorts.

Files:
- `models/Hickey/GSD2a.py` - Core Hickey gait sequence detection class
- `models/Hickey/Hickey_all.py` - Run Hickey pipeline on supported existing datasets
- `models/Hickey/Hickey_own.py` - Run robust Hickey evaluation on self-recorded datasets

### Kheirkhahan
Wrist and two-wrist fused gait sequence detection implementations inspired by Kheirkhahan et al., with robust segmentation for discontinuous timestamps.

Files:
- `models/Kheirkhahan/GSD3_test.py` - Core Kheirkhahan single-wrist detector
- `models/Kheirkhahan/GSD3_fused.py` - Activity-level fusion of two wrists
- `models/Kheirkhahan/MM_own_all_robust.py` - Robust evaluation across QSense and free-living datasets

### Real-Time Simulation
Streaming-style inference scripts to simulate deployment behavior for single-wrist and fused two-wrist workflows.

Files:
- `models/realtime/detect_per_wrist.py` - Label-free single-wrist real-time detector
- `models/realtime/detect_fused.py` - Label-free fused two-wrist real-time detector
- `models/realtime/evaluate_per_wrist.py` - Metric evaluation for single-wrist simulation
- `models/realtime/evaluate_fused.py` - Metric evaluation for fused simulation

## Configuration

### Paths
All dataset and output paths are defined in `config/paths.py`. The system automatically detects the project root from any subdirectory, so paths work regardless of where you run scripts from.

### Hyperparameters
Model hyperparameters are centralized in `config/hyperparameters.py` (create if needed).

## Development Notes

### Adding New Datasets
1. Place dataset in `Datasets/<dataset_name>/`
2. Add path constant in `config/paths.py`
3. Create evaluation script (if needed) in `analysis/`

### Extending Models
- Keep model code in `models/{ElderNet,StrokeNet,Hickey,Kheirkhahan,realtime,mstraczkiewicz}/`
- Shared utilities → `utils/`
- Outputs automatically → `outputs/`


## References

- WISDM Activity Recognition Dataset: http://www.cis.fordham.edu/wisdm/
- HMP Fall Detection Dataset: https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition+Data+Set
- WearGait Dataset: https://physionet.org/content/weargait/1.0.0/
- Bioclite Dataset: https://zenodo.org/records/14623732
- ElderNet Repo: https://github.com/yonbrand/ElderNet
- MStraczkiewicz find_walking repo: https://github.com/MStraczkiewicz/find_walking/tree/main

## Authors

- Hendrik Böckenhauer, h.q.bockenhauer@student.tudelft.nl @hbockenhauer
- Tatiana Orlovskaia, t.orlovskaia@student.tudelft.nl @tatiana-8501
