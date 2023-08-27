
import numpy as np
import matplotlib.pyplot as plt

## 1
x = np.random.rand(100,1)
y = 2.0 + 5*x*x + 0.01*np.random.randn(100,1)

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

## 2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(x)
clf = LinearRegression()
clf.fit(X,y)

y_predicted = clf.predict(X)

## 3
from sklearn.metrics import mean_squared_error, r2_score

print(f"MSE: {mean_squared_error(y, y_predicted)}")
print(f"R2: {r2_score(y, y_predicted)}")