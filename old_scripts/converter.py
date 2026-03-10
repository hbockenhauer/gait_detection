import csv

file_path = r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed\test3_Tanya\s2_2LW.txt"

with open(file_path, newline='') as infile:
    reader = csv.reader(infile, delimiter=',')
    rows = list(reader)

with open(file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(rows)

print(f"Done -> {file_path}")