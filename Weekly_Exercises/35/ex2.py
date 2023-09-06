
import numpy as np
import matplotlib.pyplot as plt

## 1
x = np.random.rand(100,1)
y = 2.0 + 5*x*x + 0.1*np.random.randn(100,1)

n = x.size
p = 3

X = np.zeros((n,p))

for j in range(p):
    X[:,j] = x[:,0]**j

XT = np.transpose(X)
XTX = np.matmul(XT, X)

invXTX = np.linalg.inv(XTX)

beta = np.matmul(np.matmul(invXTX, XT), y)
beta0, beta1, beta2 = beta[:,0]

print(f"Own model: beta = ({beta0}, {beta1}, {beta2})")

## 2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(x)
clf = LinearRegression()
clf.fit(X,y)

y_predicted = clf.predict(X)
print(f"Using scikitlearn: beta = ({clf.intercept_}, {clf.coef_[0,1]}, {clf.coef_[0,2]})")

## 3
from sklearn.metrics import mean_squared_error, r2_score

print(f"MSE: {mean_squared_error(y, y_predicted)}")
print(f"R2: {r2_score(y, y_predicted)}")

"""
$ python3 ex2.py
Own model: beta = (1.9262688970273802, 0.3540883301600106, 4.679167012929025)
Using scikitlearn: beta = ([1.9262689], 0.35408833016009833, 4.679167012928938)
MSE: 0.010589574528173303
R2: 0.9957627424669891
"""