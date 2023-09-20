import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
from sklearn import linear_model
import pandas as pd

## This part makes the Franke function, just copied from the tasks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

## OLS-function, structured close to the code in the lecture notes, with some changes for our case
x = np.sort(np.random.rand(100))
y = np.sort(np.random.rand(100))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

def OLSfit(x,y,z,deg):
    X = np.zeros((len(x), int((deg+1)**2)-1)) #Design matrix
    k = 0
    for i in range(0, int(deg+1)):
        for j in range(0, int(deg+1)):
            if i==0 and j==0:
                a ="Dont add anything" #We dont add the intercept
            else:
                X[:,k] = x**i*y**j
                k+=1

    X_pandas = pd.DataFrame(X[:,:])
    X_pandas = X_pandas - X_pandas.mean()
    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X_pandas, z, test_size= 0.2, random_state=0)
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
    ztilde = X_train @ beta
    zpredict = X_test @ beta
    MSE_train = sklearn.metrics.mean_squared_error(z_train,ztilde)
    MSE_test = sklearn.metrics.mean_squared_error(z_test,zpredict)
    R2_train = sklearn.metrics.r2_score(z_train,ztilde)
    R2_test = sklearn.metrics.r2_score(z_test,zpredict)
    return MSE_train, MSE_test, R2_train, R2_test, beta

degs = np.linspace(1,5,5)
MSE_train_array = np.zeros(5)
MSE_test_array = np.zeros(5)
R2_train_array = np.zeros(5)
R2_test_array = np.zeros(5)
beta_list = [0]*5
for i in range(0,5):
    MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i], beta_list[i] = OLSfit(x,y,z,degs[i])

plt.plot(degs,MSE_train_array,label="MSE_train")
plt.plot(degs,MSE_test_array,label="MSE_test")
plt.xlabel("degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.plot(degs,R2_train_array,label="R2_train")
plt.plot(degs,R2_test_array,label="R2_test")
plt.xlabel("degree")
plt.ylabel("R2-score")
plt.legend()
plt.show()

"""
#Dont know how to plot beta
for i in range(0,5):
    beta_list[i] = np.pad(beta_list[i], (0, beta_list[-1].size()-beta_list[i].size()), 'constant')

beta_array = np.array(beta_list)
plt.plot(degs,beta_array[:][0],label=f"beta_{1}")
for i in range(0,5):
    for j in range(2*i+1,2*i+2):
        plt.plot(degs[i:],np. trim_zeros(beta_array[:][j]),label=f"beta_{j+1}")
plt.xlabel("degree")
plt.ylabel("beta")
plt.legend()
plt.show()
"""

#Ridge
def ridgefit(x,y,z,deg,lambda_val):
    X = np.zeros((len(x), int((deg+1)**2)-1)) #Design matrix
    k = 0
    for i in range(0, int(deg+1)):
        for j in range(0, int(deg+1)):
            if i==0 and j==0:
                a ="Dont add anything" #We dont add the intercept
            else:
                X[:,k] = x**i*y**j
                k+=1

    X_pandas = pd.DataFrame(X[:,:])
    X_pandas = X_pandas - X_pandas.mean()
    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X_pandas, z, test_size= 0.2, random_state=0)

    beta = (np.linalg.inv(np.add((X_train.T @ X_train), lambda_val*np.identity(int((deg+1)**2)-1)))) @ X_train.T @ z_train
    ztilde = X_train @ beta
    zpredict = X_test @ beta
    MSE_train = sklearn.metrics.mean_squared_error(z_train,ztilde)
    MSE_test = sklearn.metrics.mean_squared_error(z_test,zpredict)
    R2_train = sklearn.metrics.r2_score(z_train,ztilde)
    R2_test = sklearn.metrics.r2_score(z_test,zpredict)
    return MSE_train, MSE_test, R2_train, R2_test, beta

lambdas = np.array([0.0001,0.001,0.01,0.1,1])
MSE_train_array = np.zeros(5)
MSE_test_array = np.zeros(5)
R2_train_array = np.zeros(5)
R2_test_array = np.zeros(5)
beta_list = [0]*5

for i in range(0,5):
    MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i], beta_list[i] = ridgefit(x,y,z,5,lambdas[i])

plt.plot(np.log10(lambdas),MSE_train_array,label="MSE_train, Ridge")
plt.plot(np.log10(lambdas),MSE_test_array,label="MSE_test, Ridge")
plt.xlabel("log10lambda")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.plot(np.log10(lambdas),R2_train_array,label="R2_train, Ridge")
plt.plot(np.log10(lambdas),R2_test_array,label="R2_test, Ridge")
plt.xlabel("log10lambda")
plt.ylabel("R2-score")
plt.legend()
plt.show()

#Lasso

def Lassofit(x,y,z,deg,lambda_val):
    X = np.zeros((len(x), int((deg+1)**2)-1)) #Design matrix
    k = 0
    for i in range(0, int(deg+1)):
        for j in range(0, int(deg+1)):
            if i==0 and j==0:
                a ="Dont add anything" #We dont add the intercept
            else:
                X[:,k] = x**i*y**j
                k+=1

    X_pandas = pd.DataFrame(X[:,:])
    X_pandas = X_pandas - X_pandas.mean()
    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X_pandas, z, test_size= 0.2, random_state=0)

    clf = linear_model.Lasso(lambda_val,fit_intercept=False)
    clf.fit(X_train,z_train)
    ztilde = clf.predict(X_train)
    zpredict = clf.predict(X_test)
    MSE_train = sklearn.metrics.mean_squared_error(z_train,ztilde)
    MSE_test = sklearn.metrics.mean_squared_error(z_test,zpredict)
    R2_train = sklearn.metrics.r2_score(z_train,ztilde)
    R2_test = sklearn.metrics.r2_score(z_test,zpredict)
    return MSE_train, MSE_test, R2_train, R2_test

lambdas = np.array([0.0001,0.001,0.01,0.1,1])
MSE_train_array = np.zeros(5)
MSE_test_array = np.zeros(5)
R2_train_array = np.zeros(5)
R2_test_array = np.zeros(5)

for i in range(0,5):
    MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i] = Lassofit(x,y,z,5,lambdas[i])

plt.plot(np.log10(lambdas),MSE_train_array,label="MSE_train, Lasso")
plt.plot(np.log10(lambdas),MSE_test_array,label="MSE_test, Lasso")
plt.xlabel("log10lambda")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.plot(np.log10(lambdas),R2_train_array,label="R2_train, Lasso")
plt.plot(np.log10(lambdas),R2_test_array,label="R2_test, Lasso")
plt.xlabel("log10lambda")
plt.ylabel("R2-score")
plt.legend()
plt.show()


#code taken from lecture notes (5.4 The bias-variance trade-off)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

np.random.seed(2018)

n = 40
n_boostraps = 100
datapoints = 10000
maxdegree = 5

bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
MSE_array = np.zeros(maxdegree)
degs=np.linspace(1,maxdegree,maxdegree)

def bootstrap(data, datapoints):
    t = np.zeros(datapoints)
    n = len(data)
    # non-parametric bootstrap
    for i in range(datapoints):
        t[i] = np.mean(data[np.random.randint(0,n,n)])

    return t, np.std(data), np.std(t) #t, bias, variance

for deg in range(1,maxdegree+1):
    X = np.zeros((len(x), int((deg+1)**2)-1)) #Design matrix
    k = 0
    for i in range(0, int(deg+1)):
        for j in range(0, int(deg+1)):
            if i==0 and j==0:
                a ="Dont add anything" #We dont add the intercept
            else:
                X[:,k] = x**i*y**j
                k+=1

    X_pandas = pd.DataFrame(X[:,:])
    X_pandas = X_pandas - X_pandas.mean()
    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X_pandas, z, test_size= 0.2, random_state=0)
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
    ztilde = X_train @ beta
    zpredict = X_test @ beta
    MSE_train = sklearn.metrics.mean_squared_error(z_train,ztilde)
    MSE_array[deg-1] = sklearn.metrics.mean_squared_error(z_test,zpredict)

    t, bias[deg-1], variance[deg-1] = bootstrap(z_test, datapoints)

plt.plot(degs,bias,label="bias")
plt.plot(degs,variance,label="variance")
plt.plot(degs,MSE_array, label="MSE")
plt.xlabel("Degree")
plt.legend()
plt.show()



"""
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)


for degree in range(maxdegree):
    X = np.zeros((len(x), int((degree+1)**2)-1)) #Design matrix
    k = 0
    for i in range(0, int(degree+1)):
        for j in range(0, int(degree+1)):
            if i==0 and j==0:
                a ="Dont add anything" #We dont add the intercept
            else:
                X[:,k] = x**i*y**j
                k+=1
    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X, z, test_size= 0.2, random_state=0)
    z_train_mean = np.mean(z_train)
    X_train_mean = np.mean(X_train,axis=0)
    X_train = X_train - X_train_mean
    z_train = z_train - z_train_mean
    z_test = z_test - z_train_mean
    z_pred = np.empty((z_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        X_, z_ = resample(X_train, z_train)
        beta = np.linalg.inv(X_.T @ X_) @ X_.T @ z_
        z_pred[:, i] = X_test @ beta

    polydegree[degree] = degree
    print(np.shape(z_test),np.shape(z_pred))
    error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
"""
