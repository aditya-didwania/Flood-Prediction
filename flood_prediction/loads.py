# stacked generalization with linear meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import tensorflow as tf
dataset = pd.read_csv('updated_test.csv')
dataset = dataset[ dataset['YEAR']>1980 ]
dataset = dataset.dropna()
X = dataset.iloc[:,[0,3,4,6,7]].values
y = dataset.iloc[:,5].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
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

# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'main_models/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = svm.SVC(kernel='linear')
	model.fit(stackedX, inputy)
	return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
    testy_enc = to_categorical(y_test)
    print(y_test.shape)
    print(testy_enc.shape)
    _, acc = model.evaluate(X_test, testy_enc, verbose=0)
    print('Model Accuracy: %.3f' % acc)
# fit stacked model using the ensemble
def init():
	model = fit_stacked_model(members, X_test, y_test)
	graph = tf.compat.v1.get_default_graph()
	print("returning")
	return model,graph
# evaluate model on test set
#yhat = stacked_prediction(members, model, X_test)
#acc = accuracy_score(y_test, yhat)
#print('Stacked Test Accuracy: %.3f' % acc)
