import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#from multimob.GSD.GSD3 import KheirkhahanGSD
from GSD2_test import HickeyGSD
from GSD3_test import KheirkhahanGSD
import matplotlib.pyplot as plt

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed\test4_Hendrik"
file_name = "s2_2LW.txt"
SAMPLING_RATE = 50 
DEBUG = False; 