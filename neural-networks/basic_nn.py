import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt

#data = pd.read_csv('data/data_onehotted_NA_removed.csv')
#data = data.drop(['INIT_DATE'], axis = 1)
data = pd.read_csv('data/data.csv')
data = shuffle(data)

X = data.iloc[:,1:-3].values
n,d_feature = X.shape
Y = data.iloc[:,-2:-1].values

X, X_test, Y, Y_test = train_test_split(X,Y,stratify = Y, test_size = 0.2)
#class_weights = {0:1., 1:1., 2:4., 3:4.}
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,stratify = Y,test_size = 0.2)

# Create one hot encoded labels
ohe = OneHotEncoder()
y_train = ohe.fit_transform(Y_train).toarray()
y_test = ohe.fit_transform(Y_test).toarray()
Y_val = ohe.fit_transform(Y_val).toarray()
n_train, d_label = y_train.shape

model = Sequential()
model.add(Dense(128, input_dim = d_feature, activation = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(d_label, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data = (X_val,Y_val),epochs=20,batch_size= 64)

y_pred = model.predict(X_test)
pred = np.argmax(y_pred,1)
val_test = np.argmax(y_test,1)
a = accuracy_score(pred,val_test)

