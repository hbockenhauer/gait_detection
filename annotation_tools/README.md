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
TXT-based annotation tool for QSense_data_mixed dataset.

**Usage:**
```bash
python annotation_tools/mixed_annotater.py
```

**Purpose:**
- Annotate mixed subject QSense data
- Update activity labels in txt format

**Expected Datasets:**
- `Datasets/QSense_data_mixed/`

**To annotate new data:**
- Change the path to desired folder, the script will look for files name `s1_1RW.txt`, `s2_2LW.txt` and `s3_3RL.txt`in that folder.
- (Optional) You can input the beginning and the end time, the file will be cropped to that time range, for example 
```bash
TIME_RANGE = (time(12, 5, 10), time(12, 10, 50))
```
- Adjust the `TIMESTAMPS` variable with the times at which the condition changes. Use the same format as before, eg. `time(12, 5, 10)` 
- Adjust the `LABELS` variable with the values you want to be assigned to the period inbetween the start to first timestamp, first to second timestamp, etc. (there should be one more label than there are timestamps)
- The script creates a new file of the same data with the new `Label` column. This file is names after the original, while the original is named with `_old.txt` at the end. 


## Output Format

Both tools modify the provided files (CSV or txt respectively) to add a column "Label" with 1 assigned to the periods of walking and 0 otherwise. 

## Adding New Annotation Tools

1. Create script in `annotation_tools/`
2. Add import to `__init__.py`
3. Document usage in this README
