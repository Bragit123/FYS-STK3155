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
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import scikitplot as skplt

data = load_breast_cancer()

class Boosting: #mangler å brukte y_pred på test dataen

    def __init__(self,data,max_depth: int,  n_estimators: int,learning_rate: float, algorthim: str ) -> None:
        self.max_depth = max_depth
        self.data = data
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorthim = algorthim
        self.X_train, self.X_test,self.y_train,self.y_test = train_test_split(data.data,data.target, test_size=0.2)

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)


 #AdaBoost (Adaptive Boosting), Gradient Boosting, and XGBoost.

    def AdaBoost(self):

        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.max_depth), n_estimators=self.n_estimators,
                                      algorithm=self.algorthim, learning_rate=self.learning_rate, random_state=42)

        

        #dont scale y_data
        ada_clf.fit(self.X_train_scaled, self.y_train)

        y_pred = ada_clf.predict(self.X_test_scaled)

        return y_pred
    

    def Gradient_Boosting(self): #change to classification

        model = GradientBoostingRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, learning_rate= self.learning_rate)  
        model.fit(self.X_train_scaled,self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        
        return y_pred
    

    def XGBoost(self,maxdegree):

        error = np.zeros(maxdegree)
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)
        polydegree = np.zeros(maxdegree)
        predictions = np.zeros(maxdegree)

        for degree in range(maxdegree):
            model =  xgb.XGBRegressor(objective ='reg:squarederror', colsaobjective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = degree, alpha = 10, n_estimators = 200)

            model.fit(self.X_train_scaled,self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            predictions[degree] = y_pred

            polydegree[degree] = degree
            error[degree] = np.mean( np.mean((self.y_test - y_pred)**2) )
            bias[degree] = np.mean( (self.y_test - np.mean(y_pred))**2 )
            variance[degree] = np.mean( np.var(y_pred) )

            print('Max depth:', degree)
            print('Error:', error[degree])
            print('Bias^2:', bias[degree])
            print('Var:', variance[degree])
            print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

        return predictions
                        






skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

plt.show()
y_probas = ada_clf.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_probas)
plt.show()

skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()
