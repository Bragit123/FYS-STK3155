import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scheduler import *
from funcs import *
from random import random, seed
import pandas as pd
import jax.numpy as jnp
from jax import grad, random, jit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import sklearn
from sklearn.model_selection import train_test_split
from plotting import *

def gd(n_iter,scheduler,grad_func,lmb=0):
    beta = random.normal(key,shape=(deg+1,1))
    for i in range(n_iter):
        gradient = grad_func(beta,y_train,X_train,lmb)
        beta -= scheduler.update_change(gradient)
    return beta

def SGD(M, n_epochs, scheduler, grad_func,lmb=0):
    beta = random.normal(key,shape=(deg+1,1))
    m = int(n/M) #number of minibatches
    n_epochs = n_iter
    for epoch in range(n_epochs):
        for iter in range(1, m+1):
            random_index = M*random.randint(key,shape=(1,),minval=0,maxval=m)[0]
            Xi = X_train[random_index:random_index+M]
            yi = y_train[random_index:random_index+M]
            gradient = (1/M)*grad_func(beta,yi,Xi,lmb)
            beta -= scheduler.update_change(gradient)
    return beta

def f(x):
    return 5.0*x**2 + 3.0*x + 1.0

key = random.PRNGKey(123)
# the number of datapoints
n = 100
x = np.linspace(0,1,100)
x = x.reshape(-1, 1)
y = f(x)
y += 0.1*random.normal(key,shape=(n,1)) #Added noise
##print(x)
#plt.plot(x,y,label = "Function")
#plt.scatter(x,y, 5, label="With added noise")

deg = 2
poly = PolynomialFeatures(degree=deg)
X = poly.fit_transform(x) # Find feature matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

XT_X = X.T @ X
beta_linreg = jnp.linalg.inv(X.T @ X) @ (X.T @ y)
print("beta from OLS")
print(beta_linreg)
#y_OLS = beta_linreg[0] + beta_linreg[1]*x + beta_linreg[2]*x**2
#plt.plot(x,y_OLS, label="Fit with OLS")

def CostRidge(beta,y,X,lmb):
    return jnp.sum((y-X @ beta)**2) + jnp.sum(lmb*beta**2)
def CostOLS(beta,y,X,lmb=0):
    return jnp.sum((y-X @ beta)**2)


##Testing different etas and lambdas
eta_vals = np.logspace(-4,-2,3)
lmbd_vals = np.logspace(-2,0,3)

grad_func = grad(CostRidge)
MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        scheduler = Momentum(eta=eta_vals[i], momentum=0.001)
        beta = SGD(M, n_epochs, scheduler, grad_func, lmbd_vals[j])
        y_mom = X_test@beta #beta[0] + beta[1]*x_test + beta[2]*x_test**2
        MSE[i,j] = sklearn.metrics.mean_squared_error(y_test,y_mom)
        R2[i,j] = sklearn.metrics.r2_score(y_test, y_mom)
        #print(R2)

heatmap(MSE, xticks=lmbd_vals, yticks=eta_vals, title="MSE test, SGD with momentum", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/GDMSEmom.pdf")
heatmap(R2, xticks=lmbd_vals, yticks=eta_vals, title="R2-score, SGD with momentum", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/GDR2mom.pdf")

#OLS as cost function
MSEs_OLS = np.zeros(16) #16 methods to be tested, we do not include GD and GD with mom. in the plot
R2s_OLS =  np.zeros(16)
MSEs_Ridge = np.zeros(16)
R2s_Ridge = np.zeros(16)
Names = ["GD","GD, mom.","SGD", "SGD, mom.","Adagrad", "Adagrad, mom.", "Adagrad, SGD", "Adagrad, SGD, mom.", "RMS-prop",
        "RMS-prop, mom.", "RMS-prop, SGD", "RMS-prop, SGD, mom.", "Adam", "Adam, mom.", "Adam, SGD", "Adam, SGD, mom."]
GD_or_SGD = ["GD","GD","SGD","SGD","GD","GD","SGD","SGD","GD","GD","SGD","SGD","GD","GD","SGD","SGD"]
schedulers = [Constant(eta=0.01), Momentum(eta=0.01, momentum=0.001),Constant(eta=0.01),Momentum(eta=0.01, momentum=0.001),
                Adagrad(eta=0.01),AdagradMomentum(eta=0.01, momentum=0.001),Adagrad(eta=0.01),AdagradMomentum(eta=0.01, momentum=0.001),
                RMS_prop(eta=0.01, rho=0.99),RMS_propMomentum(eta=0.01, rho=0.99, momentum=0.001),RMS_prop(eta=0.01, rho=0.99),RMS_propMomentum(eta=0.01, rho=0.99, momentum=0.001),
                Adam(eta=0.01, rho=0.9, rho2=0.999), AdamMomentum(eta=0.01, rho=0.9, rho2=0.999, momentum=0.001), Adam(eta=0.01, rho=0.9, rho2=0.999), AdamMomentum(eta=0.01, rho=0.9, rho2=0.999, momentum=0.001)]
n_iter = 100
M = 20  #Batch size
n_epochs = n_iter
grad_OLS = grad(CostOLS)
grad_Ridge = grad(CostRidge)


for i in range(len(MSEs_OLS)):
    scheduler = schedulers[i]
    if GD_or_SGD[i]=="SGD":
        #OLS
        beta = SGD(M, n_epochs, scheduler, grad_OLS)
        print(f"beta from {Names[i]}, OLS as costfunction")
        print(beta)
        y_pred = X_test@beta
        MSEs_OLS[i] = sklearn.metrics.mean_squared_error(y_test,y_pred)
        R2s_OLS[i] = sklearn.metrics.r2_score(y_test, y_pred)

        #Ridge
        beta = SGD(M, n_epochs, scheduler, grad_Ridge, lmb=0.1)
        print(f"beta from {Names[i]}, Ridge as costfunction")
        print(beta)
        y_pred = X_test@beta
        MSEs_Ridge[i] = sklearn.metrics.mean_squared_error(y_test,y_pred)
        R2s_Ridge[i] = sklearn.metrics.r2_score(y_test, y_pred)
    else:
        #OLS
        beta = gd(n_iter,scheduler,grad_OLS)
        print(f"beta from {Names[i]}, OLS as costfunction")
        print(beta)
        y_pred = X_test@beta
        MSEs_OLS[i] = sklearn.metrics.mean_squared_error(y_test,y_pred)
        R2s_OLS[i] = sklearn.metrics.r2_score(y_test, y_pred)

        #Ridge
        beta = gd(n_iter,scheduler,grad_Ridge, lmb=0.1)
        print(f"beta from {Names[i]}, Ridge as costfunction")
        print(beta)
        y_pred = X_test@beta
        MSEs_Ridge[i] = sklearn.metrics.mean_squared_error(y_test,y_pred)
        R2s_Ridge[i] = sklearn.metrics.r2_score(y_test, y_pred)

MSEs_OLS = MSEs_OLS[2:] #16 methods to be tested, we do not include GD and GD with mom. in the plot, because they give bad results
R2s_OLS =  R2s_OLS[2:]
MSEs_Ridge = MSEs_Ridge[2:]
R2s_Ridge = R2s_Ridge[2:]
Names = Names[2:]
#OLS
barplot(Names, MSEs_OLS, xlabel = "Method", ylabel = "MSE", title = "MSE error for different GD methods", filename="../Figures/GDMSEcostols.pdf") #We made plotting functions
barplot(Names, R2s_OLS, xlabel = "Method", ylabel = "R2", title = "R2-score for different GD methods", filename="../Figures/GDR2costols.pdf")
#Ridge
barplot(Names, MSEs_Ridge, xlabel = "Method", ylabel = "MSE", title = "MSE error for different GD methods", filename="../Figures/GDMSEcostridge.pdf") #We made plotting functions
barplot(Names, R2s_Ridge, xlabel = "Method", ylabel = "R2", title = "R2-score for different GD methods", filename="../Figures/GDR2costridge.pdf")
