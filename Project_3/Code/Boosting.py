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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from pydot import graph_from_dot_data
import pandas as pd
import os
import scikitplot as skplt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

from tensorflow.keras import datasets


class Boosting: #mangler å brukte y_pred på test dataenls

    def __init__(self, X_train, y_train, X_test, y_test, max_depth: int,  n_estimators: int,learning_rate: float, algorthim: str = "SAMME.R" ) -> None:

        self.max_depth = max_depth
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorthim = algorthim

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)


 #AdaBoost (Adaptive Boosting), Gradient Boosting, and XGBoost.

    def AdaBoost(self):

        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=self.n_estimators,
                                      algorithm=self.algorthim, learning_rate=self.learning_rate, random_state=42)

        

        #dont scale y_data
        ada_clf.fit(self.X_train_scaled, self.y_train)

        y_pred = ada_clf.predict(self.X_test_scaled)

        accuracy = np.mean(y_pred == self.y_test)

        y_probas = ada_clf.predict_proba(self.X_test)


        #print(f"Accuracy:{np.mean(y_pred == self.y_test)}")
        #print(f"self.y_test. {len(self.y_test)}")
        #print(f"{len(y_pred)}")

        #print(y_probas)

        return y_pred,y_probas, accuracy
    

    def Gradient_Boosting(self): #change to classification

        #model = GradientBoostingRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, learning_rate= self.learning_rate)

        model = GradientBoostingClassifier(max_depth = self.max_depth, n_estimators=self.n_estimators, learning_rate=self.learning_rate)  

        model.fit(self.X_train_scaled,self.y_train)
        y_pred = model.predict(self.X_test_scaled)

        accuracy = np.mean(y_pred == self.y_test)

        print(f"y_pred: {y_pred}")

        #print(f"accuracy:{accuracy} ")

        y_probas = model.predict_proba(self.X_test)
        
        return y_pred,y_probas, accuracy
    

    def XGBoost(self,maxdegree):

        error = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)
        polydegree = np.zeros(maxdegree)
        predictions = list()
        accuracy = np.zeros(maxdegree)

        for degree in range(maxdegree):
            
            #model =  xgb.XGBRegressor(objective ='reg:squarederror', colsaobjective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = self.max_depth, alpha = 10, n_estimators = 200)
            model =  xgb.XGBClassifier(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                                       max_depth = self.max_depth, alpha = 10, n_estimators = 200)

            #should max depth be equal to degree ? 

            model.fit(self.X_train_scaled,self.y_train)

            #get no changes in accuracy for higher degrees 

            y_pred = model.predict(self.X_test_scaled)
            #print(y_pred)

            y_probas = model.predict_proba(self.X_test)

            predictions.append(y_pred)

            accuracy[degree] = np.mean(y_pred == self.y_test)

            polydegree[degree] = degree

            error[degree] = np.mean(np.mean((self.y_test - y_pred)**2))

            bias[degree] = np.mean((self.y_test - np.mean(y_pred))**2)

            variance[degree] = np.mean(np.var(y_pred))

            print('Max depth:', degree)

            print('Error:', error[degree])
            print('Bias^2:', bias[degree])
            print('Var:', variance[degree])
        
            print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree],  bias[degree] + variance[degree]))

        print(accuracy)
        print(f"Higest accuray:{np.max(accuracy)} for Degree:{np.argmax(accuracy)}")

        return predictions, y_probas
    
    def cumulative_gain(self):

        #y_probas = self.Gradient_Boosting()[0]
        y_probas = self.AdaBoost()[1]
        skplt.metrics.plot_cumulative_gain(self.y_test, y_probas, title='Cumulative Gains Curve AdaBoost')
        y_probas = self.Gradient_Boosting()[1]
        skplt.metrics.plot_cumulative_gain(self.y_test, y_probas, title='Cumulative Gains Curve Gradient Boost')
        y_probas = self.XGBoost(6)[1]
        skplt.metrics.plot_cumulative_gain(self.y_test, y_probas, title='Cumulative Gains Curve XGBoost')
        plt.show()

    def ROC(self):

        skplt.metrics.plot_roc(self.y_test, self.AdaBoost()[1], title='ROC Curves AdaBoost')
        skplt.metrics.plot_roc(self.y_test, self.Gradient_Boosting()[1], title='ROC Curves Gradient Boost')
        skplt.metrics.plot_roc(self.y_test, self.XGBoost(6)[1], title='ROC Curves XGBoost')

        plt.show()


    def Descision_Tree(self):

        # dataframe = pd.DataFrame(self.data.data, columns=self.data.feature_names)

        # print(dataframe)

        # y = pd.Categorical.from_codes(self.data.target, self.data.target_names)
        # y = pd.get_dummies(y)

        # print(y)

        # X_train, X_test, y_train, y_test = train_test_split(dataframe, y,
        # random_state=1)
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        tree_model = DecisionTreeClassifier(max_depth=self.max_depth)
        tree_model.fit(X_train, y_train)


                        


(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

n_train, n_rows, n_cols = np.shape(X_train)
n_test, n_rows, n_cols = np.shape(X_test)
n_features = n_rows*n_cols
X_train = np.reshape(X_train, (n_train, n_features))
X_test = np.reshape(X_test, (n_test, n_features))

instance = Boosting(X_train, y_train, X_test, y_test, 3, 100, 1)


print(instance.AdaBoost()[1])
print(instance.cumulative_gain())
print(instance.XGBoost(6))
instance.ROC()
instance.Descision_Tree()



"""

skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

plt.show()
y_probas = ada_clf.predict_proba(X_test)

c
plt.show()

skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()


"""