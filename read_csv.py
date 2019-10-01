import numpy as np
import pandas as pd


df = pd.read_csv('UNSW_NB15_testing-set.csv', header=None)

df = df[1]

df.to_csv('hunghh.csv')

print(df[3])

