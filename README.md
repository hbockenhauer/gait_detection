# Gait Detection

Gait detection algorithm using wrist-worn Inertial Measurement Unit (IMU) data for patients with stroke.

## Project Overview

This project implements Machine Learning models (ElderNet/StrokeNet) and Signal Processing algorithms (mstraczkiewicz) for gait detection from wearable IMU sensors. The models are evaluated across multiple public datasets (WISDM, HMP, WearGait, Bioclite) as well as stroke patient cohorts and self-recorded data using QSense Motion Capture IMUs.

## Directory Structure

```
gait_detection/
├── config/                    # Configuration management and centralized paths
│   ├── paths.py              # Dataset & model path definitions
│   └── hyperparameters.py    # Model hyperparameter configs
│
├── models/                    # Model implementations
│   ├── ElderNet/             # Gait detection model inference source code
│   ├── StrokeNet/            # Fine-tuned variant for stroke patients
│   └── mstraczkiewicz/       # Signal Processing MATLAB implementations
│
├── Datasets/                  # All training and evaluation data
│   ├── HMP_Dataset/          # Fall detection dataset (32Hz)
│   ├── QSense_data/          # Self-recorded data (50Hz)
│   ├── QSense_data_edge/     # Self-recorded edge cases (50 Hz)
│   ├── QSense_data_mixed/    # Self-recorded mixed activity (50 Hz)
│   ├── WearGait/             # Parkinson's and age-matched control patients (100 Hz)
│   ├── wisdm-dataset/        # Public activity recognition dataset (20Hz)
│   ├── Free_living/          # Stroke patient data (50 Hz)
│   └── Bioclite/             # 6-activity reference dataset (50 Hz)
│
├── analysis/                  # Comparative analysis and evaluation scripts
│   ├── cross_dataset.py      # Multi-dataset evaluation
│   └── leg_classification.py # Leg classification analysis using QSense algorithm
│
├── annotation_tools/          # Data annotation and validation tools
│   ├── video_annotater.py    # Video frame labeling tool for Free_living data
│   └── mixed_annotater.py    # Annotation tool for QSense_data_mixed files
│
├── utils/                     # Shared utilities and helpers
│   ├── visualization.py      # Plotting and visualization functions
│   ├── data_loaders.py       # Common dataset loading utilities
|	├── plot_ROC_PR.py		  # Plot ROC curve and precision-recall curve
│	├── plot_qsense_activities.py # Plot QSense data with signal processing characteristics
|	└── comp_load.py 		  # Script to compute compuational load of model
|
├── outputs/                   # Generated outputs
    ├── plots/                # Visualizations and result plots
    ├── results/              # CSV results and metrics
    └── logs/                 # Training and evaluation logs

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
Core gait detection model trained on multiple public datasets. Supports inference on provided datasets.

Files:
- `models/ElderNet/eldernet_*.py` - Dataset-specific evaluation scripts
- `models/ElderNet/eldernet_unified.py` - Multi-dataset unified interface

### StrokeNet
Fine-tuned variant of ElderNet optimized for stroke patients. Used for activity-level metric computation.

Files:
- `models/StrokeNet/strokenet.py` - Main inference pipeline
- `models/StrokeNet/retrain_eldernet.py` - Fine-tuning script
- `models/StrokeNet/strokenet_utils.py` - Script to run necessary data loaders and plotters for inference

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
- Bioclite Dataset: https://zenodo.org/records/14623732
- ElderNet Repo: https://github.com/yonbrand/ElderNet
- MStraczkiewicz find_walking repo: https://github.com/MStraczkiewicz/find_walking/tree/main

## Authors

- Hendrik Böckenhauer, h.q.bockenhauer@student.tudelft.nl @hbockenhauer
- Tatiana Orlovskaia, t.orlovskaia@student.tudelft.nl @tatiana-8501
