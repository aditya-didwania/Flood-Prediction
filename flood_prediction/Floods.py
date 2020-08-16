# -*- coding: utf-8 -*-


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('updated_test.csv')
dataset = dataset[ dataset['YEAR']>1980 ]
#dataset = dataset[ dataset['SUBDIVISION'] != 'RAYALSEEMA']
#dataset = dataset[ dataset['SUBDIVISION'] != 'KUCH']
#dataset = dataset[ dataset['SUBDIVISION'] != 'LAKSHWADEEP']
#dataset = dataset[ dataset['SUBDIVISION'] != 'VIDARBHA']
dataset = dataset.dropna()

#without climate
X = dataset.iloc[:,[0,3,4,6]].values
y = dataset.iloc[:,5].values

dataset2 = dataset[dataset['SEVERITY']!='0']

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
# Encoding the Dependent Variable


#For Neural network

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



#without subdivision

X = dataset.iloc[:,[3,4,6,7]].values
y = dataset.iloc[:,5].values

dataset2 = dataset[dataset['SEVERITY']>0]

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = np.reshape(y_train,(-1,1))
y_train = onehotencoder.fit_transform(y_train).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
#normalization
#val-mean/n
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


"""
#using decision tree
# Fitting Decision Tree Classification to the Training set"""
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#-p+logp+-p-logp-
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)


#Using neural nets
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = 53))
#uniform,zero,normal
# Adding the second hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#adam,sgd,adagrad,adadelta,adamax
#MSL,MAE,MBE,crossentropy
#accuracy,probabalistic,regression,classification(t/f)
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 500)

#Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
res = np.argmax(y_pred, axis=1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, res)


#Saving the model
# serialize model to JSON
model_json = classifier.to_json()
with open("./model/flood_model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#hierarhchical data format
classifier.save_weights("./model/flood_model1.h5")
print("Saved model to disk")
