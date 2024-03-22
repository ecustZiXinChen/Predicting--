import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_excel('dataILnew.xlsx')
y_cosmo = pd.DataFrame(df.iloc[:, -1])
print(y_cosmo)
yexp = pd.DataFrame(df.iloc[:, -2])
print(yexp)
r2 = r2_score(yexp, y_cosmo)
mae = mean_absolute_error(yexp, y_cosmo)
print(r2, mae)
