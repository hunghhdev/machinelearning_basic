import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

data = pd.read_csv('mymissdata.csv', header=None)
print(data)
X = data.values
imp = Imputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(X)
result = imp.transform(X)
print('ket qua la: ')
print(result)
