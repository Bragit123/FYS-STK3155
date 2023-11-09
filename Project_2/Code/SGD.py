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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#print(x)
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
y_OLS = beta_linreg[0] + beta_linreg[1]*x + beta_linreg[2]*x**2
plt.plot(x,y_OLS, label="Fit with OLS")

def CostRidge(beta,y,X,lmb):
    return jnp.sum((y-X @ beta)**2) + jnp.sum(lmb*beta**2)

n_iter = 100
M = 5  #Batch size
n_epochs = n_iter
##Testing different etas and lambdas
eta_vals = np.logspace(-3,-1,3)
lmbd_vals = np.logspace(-3,-1,3)

grad_func = grad(CostRidge)
MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        scheduler = Momentum(eta=eta_vals[i], momentum=0.001)
        beta = SGD(M, n_epochs, scheduler, grad_func, lmbd_vals[j])
        y_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
        MSE[i,j] = sklearn.metrics.mean_squared_error(y_test,y_mom)
        R2[i,j] = sklearn.metrics.r2_score(y_test, y_mom)

heatmap(MSE, xticks=lmbd_vals, yticks=eta_vals, title="MSE test, SGD with momentum", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/GDMSEmom.pdf")
heatmap(R2, xticks=lmbd_vals, yticks=eta_vals, title="R2-score, SGD with momentum", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/GDR2mom.pdf")

#OLS as cost function
MSEs = np.zeros(16) #16 methods to be tested
R2s =  np.zeros(16)
Names = [""]*16
#plt.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)[source]

k = 0 #Counter
def CostOLS(beta,y,X,lmb=0):
    return jnp.sum((y-X @ beta)**2)

grad_func = grad(CostOLS)
#Constant learning schedule
scheduler = Constant(eta=0.001)
beta = gd(n_iter,scheduler,grad_func)
print("beta from GD")
print(beta)
y_const_eta = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_const_eta)
R2s[k] = sklearn.metrics.r2_score(y_test, y_const_eta)
Names[k] = "GD"
k += 1

#Constant learning schedule with momentum
scheduler = Momentum(eta=0.001, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func)
print("beta from GD with mom.")
print(beta)
y_const_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_const_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_const_mom)
Names[k] = "GD, mom."
k += 1

#Using SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = Constant(eta=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func)
print("beta from SGD")
print(beta)
y_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_SGD)
R2s[k] = sklearn.metrics.r2_score(y_test, y_SGD)
Names[k] = "SGD"
k += 1

#Using SGD with momentum
M = 5  #Batch size
n_epochs = 100
scheduler = Momentum(eta=0.001, momentum=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func)
print("beta from SGD with mom.")
print(beta)
y_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_SGD_mom)
Names[k] = "SGD, mom."
k += 1

#Adagrad
scheduler = Adagrad(eta=0.001)
beta = gd(n_iter,scheduler, grad_func)
print("beta from Adagrad")
print(beta)
y_adagrad = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adagrad)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adagrad)
Names[k] = "Adagrad"
k += 1

#Adagrad with momentum
scheduler = AdagradMomentum(eta=0.001, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func)
print("beta from Adagrad with mom.")
print(beta)
y_adagrad_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adagrad_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adagrad_mom)
Names[k] = "Adagrad, mom."
k += 1

#Using Adagrad with SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = Adagrad(eta=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func)
print("beta from Adagrad with SGD")
print(beta)
y_adagrad_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_SGD_mom)
Names[k] = "Adagrad, SGD"
k += 1

#Using Adagrad with SGD and momentum
M = 5  #Batch size
n_epochs = 100
scheduler = AdagradMomentum(eta=0.001, momentum=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func)
print("beta from Adagrad with SGD and mom.")
print(beta)
y_adagrad_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adagrad_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adagrad_SGD_mom)
Names[k] = "Adagrad, SGD, mom."
k += 1

#RMS-prop
scheduler = RMS_prop(eta=0.001, rho=0.99)
beta = gd(n_iter,scheduler, grad_func)
print("beta from RMS-prop")
print(beta)
y_rms = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms)
Names[k] = "RMS-prop"
k += 1

#RMS-prop with momentum
scheduler = RMS_propMomentum(eta=0.001, rho=0.99, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func)
print("beta from RMS-prop with mom.")
print(beta)
y_rms_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms_mom)
Names[k] = "RMS-prop, mom."
k += 1

#Using RMS-prop with SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = RMS_prop(eta=0.001, rho=0.99)
beta = SGD(M, n_epochs, scheduler, grad_func)
print("beta from RMS-prop with SGD")
print(beta)
y_rms_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms_SGD)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms_SGD)
Names[k] = "RMS-prop, SGD"
k += 1

#Using RMS-prop with SGD and momentum
M = 5  #Batch size
n_epochs = 100
scheduler = RMS_propMomentum(eta=0.001, rho=0.99, momentum=0.001)
print("beta from RMS-prop with SGD and mom.")
print(beta)
y_rms_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms_SGD_mom)
Names[k] = "RMS-prop, SGD, mom."
k += 1

#Adam
scheduler = Adam(eta=0.001, rho=0.9, rho2=0.999)
beta = gd(n_iter,scheduler, grad_func)
print("beta from Adam")
print(beta)
y_adam = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam)
Names[k] = "Adam"
k += 1

#Adam with momentum
scheduler = AdamMomentum(eta=0.001, rho=0.9, rho2=0.999, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func)
print("beta from Adam with mom.")
print(beta)
y_adam_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam_mom)
Names[k] = "Adam, mom."
k += 1

#Using Adam with SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = Adam(eta=0.001, rho=0.9, rho2=0.999)
beta = SGD(M, n_epochs, scheduler, grad_func)
print("beta from Adam with SGD")
print(beta)
y_adam_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam_SGD)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam_SGD)
Names[k] = "Adam, SGD"
k += 1

#Using Adam with SGD and momentum
M = 5  #Batch size
n_epochs = 100
scheduler = AdamMomentum(eta=0.001, rho=0.9, rho2=0.999, momentum=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func)
print("beta from Adam with SGD and mom.")
print(beta)
y_adam_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam_SGD_mom)
Names[k] = "Adam, SGD, mom."
k += 1

barplot(Names, MSEs, xlabel = "Method", ylabel = "MSE", title = "MSE error for different GD methods", filename="../Figures/GDMSEcostols.pdf") #We made plotting functions
barplot(Names, R2s, xlabel = "Method", ylabel = "R2", title = "R2-score for different GD methods", filename="../Figures/GDR2costols.pdf")

#Ridge as cost function
MSEs = np.zeros(16) #16 methods to be tested
R2s =  np.zeros(16)
Names = [""]*16
#plt.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)[source]

k = 0 #Counter

grad_func= grad(CostRidge)
lmb = 0.01
#Constant learning schedule

scheduler = Constant(eta=0.001)
beta = gd(n_iter,scheduler,grad_func,lmb)
print("beta from GD")
print(beta)
y_const_eta = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_const_eta)
R2s[k] = sklearn.metrics.r2_score(y_test, y_const_eta)
Names[k] = "GD"
k += 1

#Constant learning schedule with momentum
scheduler = Momentum(eta=0.001, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func,lmb)
print("beta from GD with mom.")
print(beta)
y_const_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_const_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_const_mom)
Names[k] = "GD, mom."
k += 1

#Using SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = Constant(eta=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func,lmb)
print("beta from SGD")
print(beta)
y_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_SGD)
R2s[k] = sklearn.metrics.r2_score(y_test, y_SGD)
Names[k] = "SGD"
k += 1

#Using SGD with momentum
M = 5  #Batch size
n_epochs = 100
scheduler = Momentum(eta=0.001, momentum=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func,lmb)
print("beta from SGD with mom.")
print(beta)
y_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_SGD_mom)
Names[k] = "SGD, mom."
k += 1

#Adagrad
scheduler = Adagrad(eta=0.001)
beta = gd(n_iter,scheduler, grad_func, lmb)
print("beta from Adagrad")
print(beta)
y_adagrad = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adagrad)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adagrad)
Names[k] = "Adagrad"
k += 1

#Adagrad with momentum
scheduler = AdagradMomentum(eta=0.001, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func, lmb)
print("beta from Adagrad with mom.")
print(beta)
y_adagrad_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adagrad_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adagrad_mom)
Names[k] = "Adagrad, mom."
k += 1

#Using Adagrad with SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = Adagrad(eta=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func, lmb)
print("beta from Adagrad with SGD")
print(beta)
y_adagrad_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_SGD_mom)
Names[k] = "Adagrad, SGD"
k += 1

#Using Adagrad with SGD and momentum
M = 5  #Batch size
n_epochs = 100
scheduler = AdagradMomentum(eta=0.001, momentum=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func, lmb)
print("beta from Adagrad with SGD and mom.")
print(beta)
y_adagrad_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adagrad_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adagrad_SGD_mom)
Names[k] = "Adagrad, SGD, mom."
k += 1

#RMS-prop
scheduler = RMS_prop(eta=0.001, rho=0.99)
beta = gd(n_iter,scheduler, grad_func, lmb)
print("beta from RMS-prop")
print(beta)
y_rms = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms)
Names[k] = "RMS-prop"
k += 1

#RMS-prop with momentum
scheduler = RMS_propMomentum(eta=0.001, rho=0.99, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func, lmb)
print("beta from RMS-prop with mom.")
print(beta)
y_rms_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms_mom)
Names[k] = "RMS-prop, mom."
k += 1

#Using RMS-prop with SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = RMS_prop(eta=0.001, rho=0.99)
beta = SGD(M, n_epochs, scheduler, grad_func, lmb)
print("beta from RMS-prop with SGD")
print(beta)
y_rms_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms_SGD)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms_SGD)
Names[k] = "RMS-prop, SGD"
k += 1

#Using RMS-prop with SGD and momentum
M = 5  #Batch size
n_epochs = 100
scheduler = RMS_propMomentum(eta=0.001, rho=0.99, momentum=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func, lmb)
print("beta from RMS-prop with SGD and mom.")
print(beta)
y_rms_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_rms_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_rms_SGD_mom)
Names[k] = "RMS-prop, SGD, mom."
k += 1

#Adam
scheduler = Adam(eta=0.001, rho=0.9, rho2=0.999)
beta = gd(n_iter,scheduler, grad_func, lmb)
print("beta from Adam")
print(beta)
y_adam = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam)
Names[k] = "Adam"
k += 1

#Adam with momentum
scheduler = AdamMomentum(eta=0.001, rho=0.9, rho2=0.999, momentum=0.001)
beta = gd(n_iter,scheduler, grad_func, lmb)
print("beta from Adam with mom.")
print(beta)
y_adam_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam_mom)
Names[k] = "Adam, mom."
k += 1

#Using Adam with SGD
M = 5  #Batch size
n_epochs = n_iter
scheduler = Adam(eta=0.001, rho=0.9, rho2=0.999)
beta = SGD(M, n_epochs, scheduler, grad_func, lmb)
print("beta from Adam with SGD")
print(beta)
y_adam_SGD = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam_SGD)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam_SGD)
Names[k] = "Adam, SGD"
k += 1

#Using Adam with SGD and momentum
M = 5  #Batch size
n_epochs = 100
scheduler = AdamMomentum(eta=0.001, rho=0.9, rho2=0.999, momentum=0.001)
beta = SGD(M, n_epochs, scheduler, grad_func, lmb)
print("beta from Adam with SGD and mom.")
print(beta)
y_adam_SGD_mom = beta[0] + beta[1]*x_test + beta[2]*x_test**2
MSEs[k] = sklearn.metrics.mean_squared_error(y_test,y_adam_SGD_mom)
R2s[k] = sklearn.metrics.r2_score(y_test, y_adam_SGD_mom)
Names[k] = "Adam, SGD, mom."
k += 1

barplot(Names, MSEs, xlabel = "Method", ylabel = "MSE", title = "MSE error for different GD methods", filename="../Figures/GDMSEcostridge.pdf") #We made plotting functions
barplot(Names, R2s, xlabel = "Method", ylabel = "R2", title = "R2-score for different GD methods", filename="../Figures/GDR2costridge.pdf")
