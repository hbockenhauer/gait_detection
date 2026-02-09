import pandas as pd

df = pd.read_fwf('Verisense_Data\P001\Regular\P001_Regular.txt')
df.to_csv('P001_Regular.csv')