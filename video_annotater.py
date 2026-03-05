import os
import csv
import argparse
from datetime import time, datetime, timedelta

# =============================================================================
# CONFIGURATION — edit these before running (or use CLI arguments)
# =============================================================================

DATA_PATH = r"C:\Users\orlov\intern\gait_detection\Free_living\Device 2_sub1.csv"

# The timestamp in the recording that corresponds to t=0 seconds
# Format: "HH:MM:SS.fff"  (e.g. "13:41:50.000")
START_TIME_STR = "11:24:13.500" 

# Seconds after START_TIME at which the label switches.
# Labels alternate starting from INITIAL_LABEL.
# Example: offsets [18, 45, 50] with INITIAL_LABEL=0 produces:
#   [0, 18s) → 0   [18s, 45s) → 1   [45s, 50s) → 0   [50s, end) → 1
SWITCH_OFFSETS_SEC = [34.5, 37.6, 
                      41.4, 47.9, 
                      54.8, 60.5, 
                      66.8, 75.4, 
                      97.6, 102.8,
                      123.2, 128.0, 
                      142.6, 151.0, 
                      185.4, 189.3, 
                      194.0, 200.1, 
                      203.4, 209.6, 
                      228.9, 232.5]

# 0:34.5 - 0:37.6 
# 0:41.4 - 0:47.9 
# 0:54.8 – 1:00.5 
# 1:06.8 – 1:15.4 
# 1:37.6 – 1:42.8 
# 2:03.2 – 2:08.0 
# 2:22.6 – 2:31.0 
# 3:05.4 – 3:09.3 
# 3:14.0 – 3:20.1 
# 3:23.4 - 3:29.6 
# 3:48.9 - 3:52.5 

# Label assigned to the very first segment (before the first switch)
INITIAL_LABEL = 0

# Only annotate rows within this many seconds of START_TIME.
# Set to None to annotate the entire file.
DURATION_SEC = None  # e.g. 220

# =============================================================================
def parse_time_str(t_str: str) -> datetime:
    """
    Parse a time string into a datetime, stripping the date so comparisons
    work purely on time-of-day. Supports:
      - 'DD/MM/YYYY HH:MM:SS.fff'  (format used in the CSV data rows)
      - 'HH:MM:SS.fff'             (format used in config)
    """
    t_str = t_str.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S",
                "%H:%M:%S.%f", "%H:%M:%S"):
        try:
            dt = datetime.strptime(t_str, fmt)
            return dt.replace(year=1900, month=1, day=1)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time string: '{t_str}'")


def split_row(raw_line: str) -> list:
    """
    Split a raw line on tabs, stripping surrounding whitespace and quotes.
    Handles lines where the entire row is wrapped in a single pair of quotes,
    e.g.: '"time\tax\tay\t..."'
    """
    line = raw_line.strip().strip('"')
    return [cell.strip() for cell in line.split('\t')]


def build_label_boundaries(start_dt: datetime, offsets_sec: list, initial_label: int):
    """
    Returns a sorted list of (boundary_datetime, label_after_boundary) tuples.
    Everything before the first boundary gets initial_label.
    Label flips at each boundary.
    """
    boundaries = []
    for i, offset in enumerate(sorted(offsets_sec)):
        boundary_dt = start_dt + timedelta(seconds=offset)
        label_after = (initial_label + i + 1) % 2
        boundaries.append((boundary_dt, label_after))
    return boundaries


def get_label(row_dt: datetime, boundaries: list, initial_label: int) -> int:
    """
    Everything before the first boundary -> initial_label.
    At each boundary the label flips.
    """
    label = initial_label
    for boundary_dt, label_after in boundaries:
        if row_dt >= boundary_dt:
            label = label_after
        else:
            break
    return label


def annotate(
    data_path: str,
    start_time_str: str,
    switch_offsets_sec: list,
    initial_label: int = 0,
    duration_sec: float = None,
):
    start_dt = parse_time_str(start_time_str)
    end_dt = (start_dt + timedelta(seconds=duration_sec)) if duration_sec is not None else None

    n_switches = len(switch_offsets_sec)
    final_label = (initial_label + n_switches) % 2

    boundaries = build_label_boundaries(start_dt, switch_offsets_sec, initial_label)

    # Print a summary of the annotation plan
    print("\n=== Annotation Plan ===")
    prev_label = initial_label
    prev_time = start_dt
    for boundary_dt, seg_idx in boundaries:
        offset = (boundary_dt - start_dt).total_seconds()
        print(f"  [{prev_time.strftime('%H:%M:%S.%f')[:-3]} -> {boundary_dt.strftime('%H:%M:%S.%f')[:-3]}]  "
              f"(+{offset:.1f}s)  label = {prev_label}")
        prev_label = 1 - prev_label
        prev_time = boundary_dt
    end_str = end_dt.strftime('%H:%M:%S.%f')[:-3] if end_dt else "EOF"
    print(f"  [{prev_time.strftime('%H:%M:%S.%f')[:-3]} -> {end_str}]  label = {final_label}")
    print("=======================\n")

    base, ext = os.path.splitext(data_path)
    out_path = base + "_annotated.csv"
    backup_path = base + "_old" + ext

    rows_written = 0
    rows_skipped = 0

    with open(data_path, encoding='utf-8') as infile:
        lines = infile.readlines()

    # Row 0: metadata line (e.g. "BLE Address: ...") — skip
    # Row 1: column headers (quoted, tab-separated)
    # Row 2+: data rows (quoted, tab-separated)
    header = split_row(lines[1])
    data_lines = lines[2:]

    with open(out_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header + ['Label'])

        for raw_line in data_lines:
            if not raw_line.strip():
                continue  # skip blank lines

            row = split_row(raw_line)

            try:
                row_dt = parse_time_str(row[0])
            except (ValueError, IndexError):
                rows_skipped += 1
                continue

            # Apply time range filter
            if end_dt is not None and row_dt > end_dt:
                rows_skipped += 1
                continue

            label = get_label(row_dt, boundaries, initial_label)
            # if rows_written < 30:
                # print(label)
                # print(row_dt)
            writer.writerow(row + [label])
            rows_written += 1

        # print(boundaries)
        # print("")
        # print("init label", initial_label)
    # Rename original -> _old, annotated -> original
    # if os.path.exists(backup_path):
    #     os.remove(backup_path)
    # os.rename(data_path, backup_path)
    # os.rename(out_path, data_path)

    print(f"Done. Rows annotated: {rows_written}, rows skipped: {rows_skipped}")
    #print(f"Original backed up to: {backup_path}")
    print(f"Annotated file saved to: {out_path}")


if __name__ == "__main__":
    annotate(DATA_PATH, START_TIME_STR, SWITCH_OFFSETS_SEC, INITIAL_LABEL, DURATION_SEC)
