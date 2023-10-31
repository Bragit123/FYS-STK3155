import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import *
from numpy.random import normal, uniform
from sklearn.model_selection import train_test_split
plt.style.use("seaborn-v0_8")


# Load the terrain using sample code
terrain1 = imageio.imread('SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1', fontsize=20)
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X', fontsize=20)
plt.ylabel('Y', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("../Figures/terrainimage.pdf", bbox_inches='tight')

#Extracting the data, so we can use regression on it
N = 100
terrain1 = terrain1[:N,:N]
z_shape = np.shape(terrain1)
x = np.linspace(0,1, z_shape[0])
y = np.linspace(0,1, z_shape[1])
x, y = np.meshgrid(x, y)

z = terrain1
z = np.asarray(z)

x = x.flatten()
y = y.flatten()
z = z.flatten()

## Initiate arrays for the values that we want to compute
deg_min = 2
deg_max = 8
deg_num = deg_max-deg_min+1
degs = np.linspace(deg_min, deg_max, deg_num, dtype=int)

k = 5

## OLS
mse_ols_cv = np.zeros(deg_num)
mse_ols_cv_std = np.zeros(deg_num)
mse_ols = np.zeros(deg_num) # For OLS without crossvalidation
betas_ols = [0]*deg_num
for i in range(deg_num):
    X = FeatureMatrix(x, y, degs[i])

    # OLS with crossvalidation
    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="ols")
    mse_ols_cv[i] = mse_test_mean
    mse_ols_cv_std[i] = mse_test_std

    # OLS without crossvalidation
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)

    mse_train, mse_test, r2_train, r2_test, beta = OLSfit(X_train, X_test, z_train, z_test)
    betas_ols[i] = beta
    mse_ols[i] = mse_test

## Ridge
deg = 5
lambda_exp_start = -10
lambda_exp_stop = -3
lambda_num = 20
lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)

mse_ridge_cv = np.zeros(lambda_num)
mse_ridge_cv_std = np.zeros(lambda_num)
mse_ridge = np.zeros(lambda_num) # For Ridge without crossvalidation
betas_ridge = [0]*lambda_num

X = FeatureMatrix(x, y, deg)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)
for i in range(lambda_num):
    # Ridge and Lasso with crossvalidation
    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="ridge", lambda_val=lambdas[i])
    mse_ridge_cv[i] = mse_test_mean
    mse_ridge_cv_std[i] = mse_test_std

    # Ridge and Lasso without crossvalidation
    mse_train, mse_test, r2_train, r2_test, beta = Ridgefit(X_train, X_test, z_train, z_test, lambdas[i])
    mse_ridge[i] = mse_test
    betas_ridge[i] = beta



# Plot OLS
plt.figure()
plt.title("Mean square error of OLS regression with and without crossvalidation", fontsize=20)
plt.xlabel("Polynomial degree", fontsize=20)
plt.ylabel("Mean Square Error (MSE)", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.plot(degs, mse_ols, label="Without crossvalidation")
plt.errorbar(degs, mse_ols_cv, mse_ols_cv_std, label="With crossvalidation", capsize=5, markeredgewidth=1)
plt.legend(fontsize=20)
plt.savefig(f"../Figures/terrain_cvOLS.pdf", bbox_inches='tight')

# Plot Ridge
plt.figure()
plt.title("Mean square error of ridge regression with and without crossvalidation", fontsize=20)
plt.xlabel("Lambda", fontsize=20)
plt.ylabel("Mean Square Error (MSE)", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.plot(np.log10(lambdas), mse_ridge, label="Without crossvalidation")
plt.errorbar(np.log10(lambdas), mse_ridge_cv, mse_ridge_cv_std, label="With crossvalidation", capsize=5, markeredgewidth=1)
plt.legend(fontsize=20)
plt.savefig(f"../Figures/terrain_cvridge.pdf", bbox_inches='tight')




#Visualising models
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
x_plot = np.linspace(0,1, z_shape[0])
y_plot = np.linspace(0,1, z_shape[1])
x_plot,y_plot = np.meshgrid(x_plot,y_plot)
z_plot = np.reshape(z,z_shape)
plt.figure()
plt.title("Terrain Contour-plot", fontsize=20)
plt.contourf(x_plot,y_plot,z_plot,cmap=cm.coolwarm)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("../Figures/terraincontour.pdf", bbox_inches='tight')

#Finding the best degree:
min_mse = abs(mse_ols[0])
min_ind = 0
for i in range(len(betas_ols)):
    if abs(mse_ols[i])<min_mse:
        beta_ols = betas_ols[i]
        min_mse = abs(mse_ols[i])
        min_ind = i

print(f"Optimal degree using OLS: {int(degs[min_ind])}")
X = FeatureMatrix(x, y, degs[min_ind])

# OLS without crossvalidation
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)

mse_train, mse_test, r2_train, r2_test, beta = OLSfit(X_train, X_test, z_train, z_test)
#z_shape_test = np.array([0.2*z_shape[0], z_shape[1]], dtype=int)

zpredict = X[:,1:] @ beta
zpredict = np.reshape(zpredict,z_shape)

x_plot = np.linspace(0,1, z_shape[0])
y_plot = np.linspace(0,1, z_shape[1])
x_plot,y_plot = np.meshgrid(x_plot,y_plot)

plt.figure()
plt.title("Terrain OLS", fontsize=20)
plt.contourf(x_plot,y_plot,zpredict,cmap=cm.coolwarm)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("../Figures/terrainols.pdf", bbox_inches='tight')


#Finding the best lambda for Ridge:
min_mse = abs(mse_ridge[0])
min_ind = 0
for i in range(len(betas_ridge)):
    if abs(mse_ridge[i])<min_mse:
        beta_ridge = betas_ridge[i]
        min_mse = abs(mse_ridge[i])
        min_ind = i

deg = 5
X = FeatureMatrix(x, y, deg)
print(f"Optimal lambda using Ridge: {lambdas[min_ind]}")

# Ridge without crossvalidation
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)

mse_train, mse_test, r2_train, r2_test, beta = Ridgefit(X_train, X_test, z_train, z_test, lambdas[min_ind])
#z_shape_test = np.array([0.2*z_shape[0], z_shape[1]], dtype=int)

zpredict = X[:,1:] @ beta
zpredict = np.reshape(zpredict,z_shape)

plt.figure()
plt.title("Terrain Ridge", fontsize=20)
plt.contourf(x_plot,y_plot,zpredict,cmap=cm.coolwarm)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("../Figures/terrainridge.pdf", bbox_inches='tight')
