# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
import os 
# new_path = os.path.join(os.getcwd())
# os.chdir(new_path)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Taking care of missing dataset
from sklearn.preprocessing import Imputer
# strategy takes in values mean, median and most_frequent. axis=0 impute along columns and 
# axis=1 impute along rows
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3]) # not fitting the imputer to the entire matrix but only the specified part.
X[:,1:3] = imputer.transform(X[:,1:3]) # replacing the original data with the transformed data. 

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:,0] = label_encoder_X.fit_transform(X[:,0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)



#splitting the dataset into training and test set.
# train_test_split(*arrays, **options) takes X & y as arrays and test_size as options.
"""
value b/w 0.0 and 1.0. If int, represents absolute number of test samples. 
If none, the value is automatically set to the complement of the test size. 
"""
# random state is used for random sampling. 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # Scaler has already been fitted to X_train so, we don't need that with X_test. 


