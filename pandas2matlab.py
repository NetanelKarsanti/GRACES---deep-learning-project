import os
import pandas as pd
import scipy.io
import numpy as np


dir = r'enter your data dir '
os.chdir(dir)
dataset = os.listdir()

for df_n in dataset:
    df = pd.read_csv(df_n)
    data_dict = {'X': df.iloc[:, :-1].values, 'Y': df.iloc[:, -1].values[:, np.newaxis]}
    file_name = df_n[:10] + '.mat' # Change the name according to your data
    scipy.io.savemat(file_name, data_dict)
