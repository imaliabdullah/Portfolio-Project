"""
Created on Sun Mar 17, 2024

@author: Ali Abdullah
"""

import pandas as pd
import pickle
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# importing data
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv('assets/iris.data', names=column_names)
print(dataset.describe())

# set data for training
numpyarray = np.asarray(dataset)
X = numpyarray[:, 0:4]
y = numpyarray[:, 4]

# split the data
validation_size = 0.20
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=seed)


# spot check algorithm
models = []
models.append(("LR", LogisticRegression(solver='liblinear', multi_class='auto')))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    # 10 fold cross validation to evaluate model
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    # display the cross validation results of the current model
    names.append(name)
    results.append(cv_results)
    res = f"{name}: accuracy={cv_results.mean():.4f} std={cv_results.std():.4f}"
    print(res)

svc = SVC()
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# dump the trained model in Pickle
pickle.dump(svc, open("model.pkl", "wb"))
print("Model Dump Successfull")