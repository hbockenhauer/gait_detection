"""used in adjusting file formating in case of incorrect annotations"""

import csv
import os 

from config.paths import (
    QSENSE_CLINIC, 
    QSENSE_DATA, 
    QSENSE_EDGE, 
    QSENSE_MIXED, 
    QSENSE_TEST)

file_name = "test3_Tanya\s2_2LW.txt"
file_path = os.path.join(QSENSE_MIXED, file_name)

with open(file_path, newline='') as infile:
    reader = csv.reader(infile, delimiter=',')
    rows = list(reader)

with open(file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(rows)

print(f"Done -> {file_path}")