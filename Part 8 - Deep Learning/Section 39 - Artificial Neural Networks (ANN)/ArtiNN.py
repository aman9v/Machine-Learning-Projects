from __future__ import division
import numpy as np
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#training samples and their labels
X = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:,13].values

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x1 = LabelEncoder()
X[:,1] = label_encoder_x1.fit_transform(X[:,1])
label_encoder_x2 = LabelEncoder()
X[:,2] = label_encoder_x2.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

# splitting dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# building the network in keras
from keras.models import Sequential # to add layers to the model
from keras.layers import Dense, Activation	# lets us define the dimensions of the layers viz. no of nodes, 

model = Sequential()

"""
add first hidden layer with 6 nodes. It's a norm to take the average of the i/p layer and the o/p 
as the no. of nodes in each of the hidden layers. Also, we require an input layer and that is given by
the argument input_dim in the same declaration. 
"""
model.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# add second hidden layer with same no. of nodes as the first.
model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# add the o/p layer with just one node since the required o/p is a 0 or a 1.
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# configure the learning process of the model once it has been built. 
# metrics takes in the list of metrics to be evaluated by the model
# optimizer is the algorithm we use for finding the optimum set of weights of the model.
# loss is the loss function that is optimized by the algorithm.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
# training happens here. Batch size defines the no. of samples to be fed to the network
# before the next gradient update.
model.fit(X_train, y_train, epochs=50, batch_size=20) 

y_pred = model.predict(X_test)
y_pred = y_pred > 0.5

# making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (1550+134)/(271+45)
print(accuracy)










