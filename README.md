# Gait Detection

Gait detection algorithm using wrist-worn Inertial Measurement Unit (IMU) data for patients with stroke and healthy controls.

## Project Overview

This project implements **ElderNet** and **StrokeNet** models for activity and gait detection from wearable IMU sensors. The models are evaluated across multiple public datasets (WISDM, HMP, WearGait) and patient cohorts.

## Directory Structure

```
gait_detection/
├── config/                    # Configuration management and centralized paths
│   ├── paths.py              # Dataset & model path definitions
│   └── hyperparameters.py    # Model hyperparameter configs
│
├── models/                    # Model implementations
│   ├── ElderNet/             # Gait detection model source code
│   ├── StrokeNet/            # Fine-tuned variant for stroke patients
│   └── mstraczkiewicz/       # Reference MATLAB implementations
│
├── Datasets/                  # All training and evaluation data
│   ├── HMP_Dataset/          # Fall detection dataset (32Hz)
│   ├── QSense_data/          # Healthy control QSense data (50Hz)
│   ├── QSense_data_edge/     # Edge device variant
│   ├── QSense_data_mixed/    # Mixed subject data
│   ├── WearGait/             # Parkinson's dataset
│   ├── wisdm-dataset/        # Public activity recognition dataset (20Hz)
│   ├── Free_living/          # Naturalistic movement data
│   └── Bioclite/             # 6-activity reference dataset
│
├── analysis/                  # Comparative analysis and evaluation scripts
│   ├── cross_dataset.py      # Multi-dataset evaluation
│   └── leg_classification.py # Leg classification analysis
│
├── annotation_tools/          # Data annotation and validation tools
│   ├── video_annotater.py    # Video frame labeling tool
│   └── mixed_annotater.py    # CSV-based annotation tool
│
├── utils/                     # Shared utilities and helpers
│   ├── visualization.py      # Plotting and visualization functions
│   └── data_loaders.py       # Common dataset loading utilities
│
├── outputs/                   # Generated outputs (not tracked in git)
│   ├── plots/                # Visualizations and result plots
│   ├── results/              # CSV results and metrics
│   └── logs/                 # Training and evaluation logs
│
└── notebooks/                 # Jupyter notebooks for exploration and analysis
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

#### Running ElderNet on a dataset
```python
from config.paths import WEARGAIT_PD, PLOTS_DIR
from models.ElderNet.eldernet_WearGait import load_weargait_data, detect_sampling_rate

# Load data
data = load_weargait_data(WEARGAIT_PD)
```

#### Running Cross-Dataset Evaluation
```bash
python analysis/cross_dataset.py
```

#### Using Centralized Paths
Instead of hardcoding paths, use the centralized path configuration:
```python
from config import HMP_PATH, WEARGAIT_PD, OUTPUTS_ROOT, validate_dataset_paths

# Validate dataset setup
status = validate_dataset_paths()
```

## Models

### ElderNet
Core gait detection model trained on multiple public datasets. Supports inference on:
- WISDM (Activity Recognition)
- HMP (Fall Detection)
- WearGait (Parkinson's Disease)
- QSense variants (Clinical Testing)
- Free-living data

Files:
- `models/ElderNet/eldernet_*.py` - Dataset-specific evaluation scripts
- `models/ElderNet/eldernet_unified.py` - Multi-dataset unified interface

### StrokeNet
Fine-tuned variant of ElderNet optimized for stroke patients. Used for activity-level metric computation.

Files:
- `models/StrokeNet/strokenet.py` - Main inference pipeline
- `models/StrokeNet/retrain_eldernet.py` - Fine-tuning script

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
- Keep model code in `models/{ElderNet,StrokeNet}/`
- Shared utilities → `utils/`
- Analysis scripts → `analysis/`
- Outputs automatically → `outputs/`

### Testing
```bash
# Validate dataset setup
python -m config.paths

# Run analysis
python analysis/cross_dataset.py
```

## References

- WISDM Activity Recognition Dataset: http://www.cis.fordham.edu/wisdm/
- HMP Fall Detection Dataset: https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition+Data+Set
- WearGait Dataset: https://physionet.org/content/weargait/1.0.0/

## License

[Add license info here]

## Authors

[Add contributor info here]
