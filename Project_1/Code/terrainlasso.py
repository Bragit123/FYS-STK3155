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
N = 20
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

k = 5
deg = 5
lambda_exp_start = -10
lambda_exp_stop = -3
lambda_num = 20
lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)

mse_lasso_cv = np.zeros(lambda_num)
mse_lasso_cv_std = np.zeros(lambda_num)
mse_lasso = np.zeros(lambda_num) # For Lasso without crossvalidation
betas_lasso = [0]*lambda_num

X = FeatureMatrix(x, y, deg)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)
for i in range(lambda_num):

    mse_train_mean, mse_train_std, mse_test_mean, mse_test_std = Crossvalidation(X, z, k, model="lasso", lambda_val=lambdas[i])
    mse_lasso_cv[i] = mse_test_mean
    mse_lasso_cv_std[i] = mse_test_std

    mse_train, mse_test, r2_train, r2_test, beta = Lassofit(X_train, X_test, z_train, z_test, lambdas[i])
    mse_lasso[i] = mse_test
    betas_lasso[i] = beta
# Plot Lasso
plt.figure()
plt.title("Mean square error of lasso regression with and without crossvalidation", fontsize=20)
plt.xlabel("Lambda", fontsize=20)
plt.ylabel("Mean Square Error (MSE)", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.plot(np.log10(lambdas), mse_lasso, label="Without crossvalidation")
plt.errorbar(np.log10(lambdas), mse_lasso_cv, mse_lasso_cv_std, label="With crossvalidation", capsize=5, markeredgewidth=1)
plt.legend(fontsize=20)
plt.savefig(f"../Figures/terrain_cvlasso.pdf", bbox_inches='tight')


#Visualising models
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
#Finding the best lambda for Lasso:
min_mse = abs(mse_lasso[0])
min_ind = 0
for i in range(len(betas_lasso)):
    if abs(mse_lasso[i])<min_mse:
        beta_lasso = betas_lasso[i]
        min_mse = abs(mse_lasso[i])
        min_ind = i
print(f"Optimal lambda using Lasso: {lambdas[min_ind]}")
deg = 5
X = FeatureMatrix(x, y, deg)

# Lasso without crossvalidation
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)

mse_train, mse_test, r2_train, r2_test, beta = Lassofit(X_train, X_test, z_train, z_test, lambdas[min_ind])

zpredict = X[:,1:] @ beta
zpredict = np.reshape(zpredict,z_shape)

x_plot = np.linspace(0,1, z_shape[0])
y_plot = np.linspace(0,1, z_shape[1])
x_plot,y_plot = np.meshgrid(x_plot,y_plot)


plt.figure()
plt.title("Terrain Lasso", fontsize=20)
plt.contourf(x_plot,y_plot,zpredict,cmap=cm.coolwarm)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("../Figures/terrainlasso.pdf", bbox_inches='tight')

z_plot = np.reshape(z, z_shape)
plt.figure()
plt.title("Terrain", fontsize=20)
plt.contourf(x_plot,y_plot,z_plot,cmap=cm.coolwarm)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("../Figures/terrainlasso20x20.pdf", bbox_inches='tight')
