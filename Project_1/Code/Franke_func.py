import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from time import time
from scipy.stats import norm


np.random.seed()
n = 5 
poly_degree = 5

# Make data set. meshgrid is only for plotting 
#x = np.arange(2, 3, 0.01)
#y = np.arange(2, 3, 0.01)

x = np.random.rand(100) #need random numbers between 0 and 1, or else you get singular matrix
y = np.random.rand(100)

#Code below might be more optimal:
#x = np.sort(np.random.uniform(0, 1, N))
#y = np.sort(np.random.uniform(0, 1, N))


X, Y = np.meshgrid(x,y) #creating meashgrid. Note: When doing regression we do not want to feed our Franke function with X and Y, I could use X and Y if i ravel() 


def FrankeFunction(x,y): #defining Franke Function
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x,y) #Franke function takes in only pairs of (x,y) values, (x_1,y_1) = z_1, (x_2,y_2) = z_2, osv.
                        # If i would define z(X,Y), I.e. X, Y from meshgrid, it would pair x_1 with every y, and so on and produce 3 times more data


#feature_matrix = np.zeros((len(x), (poly_degree + 1)**2 - 1) ) #creating empty feature matrix


poly_degree_list = []
feature_matrix_list = []

for p in range(0,poly_degree):

    feature_matrix = np.zeros((len(x), (p + 2)**2 - 1) ) #creating empty feature matrix
    feature_matrix_list.append(feature_matrix)
    #print(f"feature matrix number: {p+1} = {feature_matrix_list[p]}")
    poly_degree_list.append(p + 1)
    k = 0

    for i in range(p + 2):     #nested loop, to get all combinations of x and y
        for j in range(p + 2):
            #feature_matrix = np.zeros((len(x), (poly_degree + 1)**2 - 1) ) #creating empty feature matrix
            

            #print(feature_matrix_list[p])
            if i != 0 or j !=0:                    
                feature_matrix_list[p][:,k] = (x - np.mean(x))**i * (y - np.mean(x))**j #unsure if it is correct to subtract mean value here
                k += 1

#X_train_list = []
#X_test_list = []
#z_train_list = []
#z_test_list = []

MSE_ols_train_list = []
MSE_ols_test_list = []
R2_ols_train_list = []
R2_ols_test_list = []

MSE_ridge_train_list = []
MSE_ridge_test_list = []
R2_ridge_train_list = []
R2_ridge_test_list = []

MSE_lasso_train_list = []
MSE_lasso_test_list = []
R2_lasso_train_list = []
R2_lasso_test_list = []

Identity = []

for i in range(len(feature_matrix_list)):
    X_train, X_test, z_train, z_test = train_test_split(feature_matrix_list[i], z) #splitting my data into test and train data OLS regression
    beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train #calculating beta
    z_predict_train_ols = X_train @ beta_ols
    z_predict_test_ols = X_test @ beta_ols

    MSE_ols_train_list.append(mean_squared_error(z_train,z_predict_train_ols))
    MSE_ols_test_list.append(mean_squared_error(z_test,z_predict_test_ols))
    R2_ols_train_list.append(r2_score(z_train,z_predict_train_ols))
    R2_ols_test_list.append(r2_score(z_test,z_predict_test_ols))

    lambdaa = [0.0001, 0.001, 0.01, 0.1, 1 ] #only works with polyd = 5, this gives nice figure in comparison to 2.11
    #lambdaa = [np.random.rand(1) for i in range(len(poly_degree_list)) ] 
    Identity.append(np.eye((i + 2)**2 - 1))
    beta_ridge = np.linalg.inv(X_train.T @ X_train + lambdaa[i] * Identity[i]) @ X_train.T @ z_train

    z_predict_train_ridge = X_train @ beta_ridge
    z_predict_test_ridge = X_test @ beta_ridge

    MSE_ridge_train_list.append(mean_squared_error(z_train,z_predict_train_ridge))
    MSE_ridge_test_list.append(mean_squared_error(z_test,z_predict_test_ridge))
    R2_ridge_train_list.append(r2_score(z_train,z_predict_train_ridge))
    R2_ridge_test_list.append(r2_score(z_test,z_predict_test_ridge))

    X_train_lasso, X_test_lasso, z_train_lasso, z_test_lasso = train_test_split(feature_matrix_list[i], z) #splitting my data into test and train data for Lasso regression

    model = Lasso(alpha=0.05)
    model.fit(X_train_lasso,z_train_lasso)

    z_predict_train_lasso = model.predict(X_train_lasso)
    z_predict_test_lasso = model.predict(X_test_lasso)
    

    MSE_lasso_train_list.append(mean_squared_error(z_train_lasso,z_predict_train_lasso))
    MSE_lasso_test_list.append(mean_squared_error(z_test_lasso,z_predict_test_lasso))
    R2_lasso_train_list.append(r2_score(z_train_lasso,z_predict_train_lasso))
    R2_lasso_test_list.append(r2_score(z_test_lasso,z_predict_test_lasso))




    




fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(poly_degree_list,MSE_ols_train_list, label = "MSE Train")
ax1.legend()
ax1.plot(poly_degree_list,MSE_ols_test_list, label = "MSE Test")
ax1.set_xlabel('degree ')
ax1.set_ylabel(' Error')
ax1.legend()
ax1.set_title('MSE_ols vs degree')

ax2.plot(poly_degree_list,R2_ols_train_list, label = "R2 Train")
ax2.legend()
ax2.plot(poly_degree_list,R2_ols_test_list, label = "R2 Test")
ax2.set_xlabel('degree ')
ax2.set_ylabel(' R2')
ax2.legend()
ax2.set_title('R2_ols vs degree')
plt.legend

fig2, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(poly_degree_list,MSE_ridge_train_list, label = "MSE Train")
ax1.legend()
ax1.plot(poly_degree_list,MSE_ridge_test_list, label = "MSE Test")
ax1.set_xlabel('degree ')
ax1.set_ylabel(' Error')
ax1.legend()
ax1.set_title('MSE_ridge vs degree')

ax2.plot(poly_degree_list,R2_ridge_train_list, label = "R2 Train")
ax2.legend()
ax2.plot(poly_degree_list,R2_ridge_test_list, label = "R2 Test")
ax2.set_xlabel('degree ')
ax2.set_ylabel(' R2')
ax2.legend()
ax2.set_title('R2_ridge vs degree')
plt.legend

fig3, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(poly_degree_list,MSE_lasso_train_list, label = "MSE Train")
ax1.legend()
ax1.plot(poly_degree_list,MSE_lasso_test_list, label = "MSE Test")
ax1.set_xlabel('degree ')
ax1.set_ylabel(' Error')
ax1.legend()
ax1.set_title('MSE_lasso vs degree')

ax2.plot(poly_degree_list,R2_lasso_train_list, label = "R2 Train")
ax2.legend()
ax2.plot(poly_degree_list,R2_lasso_test_list, label = "R2 Test")
ax2.set_xlabel('degree ')
ax2.set_ylabel(' R2')
ax2.legend()
ax2.set_title('R2_lasso vs degree')
plt.legend

#plt.show()






#PART E Bootstrap

def bootstrap(z):
    datapoints = 10000
    t = np.zeros(datapoints)
    n = len(z)
    # non-parametric bootstrap         
    for i in range(datapoints):
        t[i] = np.mean(z[np.random.randint(0,n,n)])
    # analysis    
    print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (np.mean(z), np.std(z),np.mean(t),np.std(t))) #Outliers are possibly eliminated from the smaller dataset t, as it is a random piece of the whole dataset z. 
    return t

# We set the mean value to 100 and the standard deviation to 15
mu, sigma = 100, 15

# bootstrap returns the data sample
# cross validation is cheaper than bootstrap                                    
t = bootstrap(z)


"""
# the histogram of the bootstrapped data (normalized data if density = True)
n, binsboot, patches = plt.hist(t, 500, facecolor='red')
# add a 'best fit' line  
y = norm.pdf(binsboot, np.mean(t), np.std(t))
lt = plt.plot(binsboot, y, 'b', linewidth=1)
plt.xlabel('z')
plt.ylabel('Probability')
plt.grid(True)

"""

#Part F: Cross-validation. Missing alot

print(len(X_test),len(z_test))
print("X",X_test,"z",z_test)

"""

plt.figure()
plt.scatter(X_test, z_test, label='Data points')
plt.legend()

"""
plt.figure()
plt.plot(poly_degree_list,MSE_ols_test_list)
plt.plot(poly_degree_list,MSE_ols_train_list)

plt.show()


