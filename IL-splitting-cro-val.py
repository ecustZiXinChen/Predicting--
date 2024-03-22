import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold

df = pd.read_excel('dataILnew.xlsx')
scaler = MinMaxScaler()
X1 = df.iloc[:, 2:16]
X1 = pd.DataFrame(X1)
print(X1)
Deviation = pd.DataFrame(df.iloc[:, -3])
print(Deviation)
yexp = pd.DataFrame(df.iloc[:, -2])
ycosmo = pd.DataFrame(df.iloc[:, -1])
Name = df.iloc[:, 1]
data = pd.concat([Name, X1, Deviation, yexp, ycosmo], axis=1)
print(data)
k = list(set(Name))
unique_names = list(set(Name))
unique_names_fixed = list(set(Name))

num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

save_directory = "10-folds"
test_names = []

for fold in range(1, 10):
    test_names = np.random.choice(unique_names, size=24, replace=False)
    test_names = list(test_names)
    for i in test_names:
        unique_names.remove(i)
    print(test_names)

    train_names = np.setdiff1d(unique_names_fixed, test_names)
    train_data = data[data['Name'].isin(train_names)]
    test_data = data[data['Name'].isin(test_names)]

    train_data.to_csv(f"{save_directory}{fold}_train.csv", index=False)
    test_data.to_csv(f"{save_directory}{fold}_test.csv", index=False)

    print(
        f"Fold {fold} - Training Data: {len(train_data)}, "
        f"Testing Data: {len(test_data)}")

print(unique_names)
fold_last = 10
test_data5 = data[data['Name'].isin(unique_names)]
train_names = np.setdiff1d(unique_names_fixed, test_data5)
train_data5 = data[data['Name'].isin(train_names)]
print(
        f"Fold {fold_last} - Training Data: {len(train_data5)}, "
        f"Testing Data: {len(test_data5)}")
train_data5.to_csv(f"{save_directory}{fold_last}_train.csv", index=False)
test_data5.to_csv(f"{save_directory}{fold_last}_test.csv", index=False)

