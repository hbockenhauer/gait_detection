# Annotation Tools

Tools for labeling and validating gait/activity data from IMU sensors.

## Tools

### `video_annotater.py`
Interactive tool for annotating video-aligned IMU data with activity labels.

**Usage:**
```bash
python annotation_tools/video_annotater.py
```

**Purpose:**
- Timestamp-based labeling of Free_living dataset
- Synchronization of video frames with accelerometer data
- Activity label assignment with frame accuracy

**Expected Datasets:**
- `Datasets/Free_living/`

### `mixed_annotater.py`
CSV-based annotation tool for QSense_data_mixed dataset.

**Usage:**
```bash
python annotation_tools/mixed_annotater.py
```

**Purpose:**
- Annotate mixed subject QSense data
- Update activity labels in CSV format
- Validate label consistency

**Expected Datasets:**
- `Datasets/QSense_data_mixed/`

## Output Format

Both tools generate annotated CSV files with columns:
- `timestamp` - Recording timestamp
- `accel_x`, `accel_y`, `accel_z` - Acceleration values
- `activity_label` - Annotated activity class

## Adding New Annotation Tools

1. Create script in `annotation_tools/`
2. Add import to `__init__.py`
3. Document usage in this README
