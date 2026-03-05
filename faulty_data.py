import csv
import os
from datetime import time


FILES = [
    r"C:\Users\orlov\intern\gait_detection\QSense_data_edge\Walking_crutches_Tanya\s1_1RW.txt",
    
    r"C:\Users\orlov\intern\gait_detection\QSense_data_edge\Walking_crutches_Tanya\s2_2LW.txt",
    #r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed\test6_Tanya\s3_3RL.txt",
]


def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)


def fix_file(file_path):
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = []
    dropped = 0
    max_time = None

    for row in rows:
        t = parse_time(row['HH:mm:ss.fff'])
        if max_time is None or t > max_time:
            max_time = t
            kept.append(row)
        else:
            dropped += 1

    # Back up original
    backup_path = file_path.replace('.txt', '_faulty.txt')
    os.rename(file_path, backup_path)

    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(kept)

    print(f"{os.path.basename(file_path)}: kept {len(kept)}, dropped {dropped} rows")


if __name__ == '__main__':
    for path in FILES:
        if os.path.exists(path):
            fix_file(path)
        else:
            print(f"Not found, skipping: {path}")
