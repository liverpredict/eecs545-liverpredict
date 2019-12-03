import pandas as pd

X_train = pd.read_csv('data/liver_data_train_final.csv')
X_test = pd.read_csv('data/liver_data_test_final.csv')

X_train.to_csv('data/x_train.csv')
X_test.to_csv('data/x_test.csv')

Y_train = pd.read_csv('data/liver_labels_train.csv')
y_180_train = Y_train.iloc[:,1:2]
y_180_train.to_csv('data/y_180_train.csv')

y_360_train = Y_train.iloc[:,2:3]
y_360_train.to_csv('data/y_360_train.csv')

y_730_train = Y_train.iloc[:,3:]
y_730_train.to_csv('data/y_730_train.csv')

Y_test = pd.read_csv('data/liver_labels_test.csv')
y_180_test = Y_test.iloc[:,1:2]
y_180_test.to_csv('data/y_180_test.csv')

y_360_test = Y_test.iloc[:,2:3]
y_360_test.to_csv('data/y_360_test.csv')

y_730_test = Y_test.iloc[:,3:]
y_730_test.to_csv('data/y_730_test.csv')