import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
from sklearn import linear_model
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

## Making the Franke function. This part is largely copied from the projection description
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Generate data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

def FrankeFunction(x, y):
    """ Calculates the Franke function at a point (x,y) """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y) # Calculate the Franke function for our dataset.

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("franke_function")


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
    z_test = z_test -np.mean(z_train)
    z_train = z_train -np.mean(z_train)
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

plt.figure()
plt.plot(degs,MSE_train_array,label="MSE_train")
plt.plot(degs,MSE_test_array,label="MSE_test")
plt.xlabel("degree")
plt.ylabel("MSE")
plt.legend()
plt.savefig("MSEOLS.png")

plt.figure()
plt.plot(degs,R2_train_array,label="R2_train")
plt.plot(degs,R2_test_array,label="R2_test")
plt.xlabel("degree")
plt.ylabel("R2-score")
plt.legend()
plt.savefig("R2OLS.png")