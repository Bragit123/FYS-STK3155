from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from functions import *

np.random.seed(200)

## Creating data set
N = 100 # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

datapoints = 100
maxdegree = 5

bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
MSE_array = np.zeros(maxdegree)
degs=np.zeros(maxdegree)

for i in range(len(degs)):
    t_x = bootstrap(x, datapoints)
    t_y = bootstrap(y, datapoints)
    t_z = bootstrap(z, datapoints)
    degs[i] = i+6
    X = np.zeros((len(t_x), int((degs[i]+1)**2)-1)) #Design matrix
    l = 0

    for j in range(0, int(degs[i])+1):
        for k in range(0, int(degs[i])+1):
            if j==0 and j==0:
                a ="Dont add anything" #We dont add the intercept
            else:
                X[:,l] = t_x**j*t_y**k
                l+=1


    X_pandas = pd.DataFrame(X[:,:])
    X_pandas = X_pandas - X_pandas.mean()
    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X_pandas, t_z, test_size= 0.2, random_state=0)
    z_test = z_test - np.mean(z_train)
    z_train = z_train - np.mean(z_train)

    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    ztilde = X_train @ beta
    zpredict = X_test @ beta
    MSE_array[i] = np.mean((z_test - zpredict)**2)
    bias[i] = np.mean((z_test - np.mean(zpredict))**2)
    variance[i] = np.var(zpredict)

plt.figure()
plt.plot(degs,bias,label="bias")
plt.plot(degs,variance,label="variance")
plt.plot(degs,MSE_array, label="MSE test")
plt.xlabel("Degree")
plt.legend()
plt.savefig("biasvariance.pdf")
