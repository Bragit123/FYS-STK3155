import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
from sklearn import linear_model

#This part makes the Franke function, just copied from the tasks

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

fig = plt.figure()
ax = fig.gca(projection='3d')

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

#OLS-function, structured close to the code in the lecture notes, with some changes for our case
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

    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X, z, test_size= 0.2, random_state=0)
    z_train_mean = np.mean(z_train)
    X_train_mean = np.mean(X_train,axis=0)
    X_train = X_train - X_train_mean
    z_train = z_train - z_train_mean
    z_test = z_test - z_train_mean
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

    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X, z, test_size= 0.2, random_state=0)
    z_train_mean = np.mean(z_train)
    X_train_mean = np.mean(X_train,axis=0)
    X_train = X_train - X_train_mean
    z_train = z_train - z_train_mean
    z_test = z_test - z_train_mean
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


    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X, z, test_size= 0.2, random_state=0)
    z_train_mean = np.mean(z_train)
    X_train_mean = np.mean(X_train,axis=0)
    X_train = X_train - X_train_mean
    z_train = z_train - z_train_mean
    z_test = z_test - z_train_mean

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
