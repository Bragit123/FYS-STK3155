from fysstkproject1regression.py import Frankefunction, OLSfit, ridgefit, Lassofit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(3155)

# Generate the data.
nsamples = 100
x = np.sort(np.random.rand(nsamples))
y = np.sort(np.random.rand(nsamples))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

## Cross-validation on Ridge regression


 #FeatureMatrix
 #OLSfit
 #ridgefit
 #Lassofit
# Initialize a KFold instance
def crossvalidation(k,x,y,z,lambda_val):
    kfold_ind = np.linspace(0,len(data), k+1)  #postition indices for us to divide
                                               #the training data and test data in different places
    X = FeatureMatrix(x,y)
    for i in range(k+1):
        X_test = X[int(kfold_ind[i]):int(kfold_ind[i+1])]
        X_train = X.splice(int(kfold_ind[i]),int(kfold_ind[i+1]))
        z_test = z[int(kfold_ind[i]):int(kfold_ind[i+1])]
        z_train = z.splice(int(kfold_ind[i]),int(kfold_ind[i+1]))

        X_test, X_train, z_test, z_train = Scale(X_test, X_train, z_test, z_train)

        MSE_train_array[i], MSE_test_array[i], R2_train, R2_test, beta = ridgefit(X_test, X_train, z_test, z_train,lambda_val)

    return np.mean(MSE_test_array)

k=5 #folds
lambdas = np.array([0.0001,0.001,0.01,0.1,1]) #lambdas = np.logspace(-3, 5, nlambdas)
MSEs_test = np.zeros(len(lambdas))
for i in range(len(lambdas)):
    MSEs_test[i] = crossvalidation(k,x,y,z,lambdas[i])

plt.figure()
plt.plot(np.log(lambdas),MSEs_test, label="MSE test")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE test")
plt.legend()
plt.savefig("crossvalridge.pdf")
