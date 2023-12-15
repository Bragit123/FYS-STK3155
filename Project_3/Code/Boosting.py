import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.metrics
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from pydot import graph_from_dot_data
import pandas as pd
import os
import plotting

import scikitplot as skplt

from tensorflow.keras import datasets


class Boosting:

    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.eta = 0.1
        self.lam = 0.1

    # AdaBoost (Adaptive Boosting), Gradient Boosting, and XGBoost.
    def AdaBoost(self):
        ada_clf = AdaBoostClassifier(learning_rate=self.eta, random_state=42)
        
        #dont scale y_data
        ada_clf.fit(self.X_train, self.y_train)
        y_pred = ada_clf.predict(self.X_test)
        accuracy = np.mean(y_pred == self.y_test)

        return accuracy
    

    def Gradient_Boosting(self):
        model = GradientBoostingClassifier(learning_rate=self.eta)

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        accuracy = np.mean(y_pred == self.y_test)
        
        return accuracy
    

    def XGBoost(self):
        model =  xgb.XGBClassifier(learning_rate = self.eta, reg_lambda = self.lam)

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        accuracy = np.mean(y_pred == self.y_test)

        return accuracy

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

n_train = int(0.1*len(y_train))
n_test = int(0.1*len(y_test))
X_train = X_train[0:n_train,:,:] ; y_train = y_train[0:n_train]
X_test = X_test[0:n_test,:,:] ; y_test = y_test[0:n_test]

n_train, n_rows, n_cols = np.shape(X_train)
n_test, n_rows, n_cols = np.shape(X_test)
n_features = n_rows*n_cols
X_train = np.reshape(X_train, (n_train, n_features))
X_test = np.reshape(X_test, (n_test, n_features))

X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0)
X_test = minmax_scale(X_test, feature_range=(0, 1), axis=0)

# Make arrays for eta and lambda
eta0 = -5; eta1 = 1; n_eta = eta1-eta0+1
lam0 = -5; lam1 = 1; n_lam = lam1-lam0+1
etas = np.logspace(eta0, eta1, n_eta)
lams = np.logspace(lam0, lam1, n_lam)

val_accs_adaboost = np.zeros(n_eta)
val_accs_gradboost = np.zeros(n_eta)
val_accs_xgboost = np.zeros((n_eta, n_lam))

instance = Boosting(X_train, y_train, X_test, y_test)

for i in range(len(etas)):
    print(f"i = {i+1} / {n_eta}")
    instance.eta = etas[i]
    val_accs_adaboost[i] = instance.AdaBoost()
    val_accs_gradboost[i] = instance.Gradient_Boosting()
    
    for j in range(len(lams)):
        print(f"j = {j+1} / {n_lam}")

        instance.lam = lams[j]
        val_accs_xgboost[i,j] = instance.XGBoost()

# Create heatmap plots for the accuracies
title_adaboost = f"Validation accuracies for Adaboost."
title_gradboost = f"Validation accuracies for Gradient Boost."
title_xgboost = f"Validation accuracies for XGBoost."
filename_adaboost = f"../Figures/val_accs_adaboost.pdf"
filename_gradboost = f"../Figures/val_accs_gradboost.pdf"
filename_xgboost = f"../Figures/val_accs_xgboost.pdf"
plotting.barplot(x=np.log10(etas), y=val_accs_adaboost, title=title_adaboost, xlabel="$\\eta$", ylabel="Accuracy", filename=filename_adaboost)
plotting.barplot(x=np.log10(etas), y=val_accs_gradboost, title=title_gradboost, xlabel="$\\eta$", ylabel="Accuracy", filename=filename_gradboost)
plotting.heatmap(data=val_accs_xgboost, xticks=lams, yticks=etas, title=title_xgboost, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename_xgboost)
