import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

data = pd.read_excel('dataILnew.xlsx')
X1 = data.iloc[:, 2:13]
X1 = pd.DataFrame(X1)
Y1 = data.iloc[:, -4]
Y1 = pd.DataFrame(Y1)

regressor = XGBRegressor(max_depth=7,
                         learning_rate=0.9,
                         n_estimators=210,
                         objective='reg:squarederror',
                         booster='gbtree',
                         gamma=0)

σ_model = regressor.fit(X1, Y1)
σ_pre = σ_model.predict(X1)
σ_pre = pd.DataFrame(σ_pre)
b = r2_score(σ_pre, Y1)
mae = mean_absolute_error(σ_pre, Y1)
print(b, mae)



