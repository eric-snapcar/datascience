# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('winequality-white.csv', sep=";")
print data.head()

def explore(X):
    fig = plt.figure(figsize=(16, 12))
    for feat_idx in range(X.shape[1]):
        ax = fig.add_subplot(3,4, (feat_idx+1))
        h = ax.hist(X[:, feat_idx], bins=50, color='steelblue',normed=True, edgecolor='none')
        ax.set_title(data.columns[feat_idx], fontsize=14)
    plt.show()
def plot(X):
    fig = plt.figure(figsize=(16, 12))
    for feat_idx in range(X.shape[1]):
        ax = fig.add_subplot(3,4, (feat_idx+1))
        h = ax.hist(X[:, feat_idx], bins=50, color='steelblue',
        normed=True, edgecolor='none')
        ax.set_title(data.columns[feat_idx], fontsize=14)
    plt.show()

X = data.as_matrix(data.columns[:-1])
y = data.as_matrix([data.columns[-1]])
y = y.flatten()
explore(X)

y_class = np.where(y<6, 0, 1)
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=0.3 )
from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

plot(X_train_std)
