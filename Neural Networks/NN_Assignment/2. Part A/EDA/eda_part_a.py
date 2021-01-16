import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

data = pd.read_csv("C:/Users/delim/Desktop/NN_Assignment/Part A/ctg_data_cleaned.csv")
profile = ProfileReport(data)
profile.to_file("cardiotocography_report.html")