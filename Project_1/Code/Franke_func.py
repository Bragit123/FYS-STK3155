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

np.random.seed()
n = 5 
poly_degree = 1

# Make data set. meshgrid is only for plotting 
#x = np.arange(2, 3, 0.01)
#y = np.arange(2, 3, 0.01)

x = np.random.rand(5) #need random numbers between 0 and 1, or else you get singular matrix
y = np.random.rand(5)



X, Y = np.meshgrid(x,y) #creating meashgrid. Note: When doing regression we do not want to feed our Franke function with X and Y, I could use X and Y if i ravel() 


def FrankeFunction(x,y): #defining Franke Function
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x,y) #Franke function takes in only pairs of (x,y) values, (x_1,y_1) = z_1, (x_2,y_2) = z_2, osv.
                        # If i would define z(X,Y), I.e. X, Y from meshgrid, it would pair x_1 with every y, and so on and produce 3 times more data


feature_matrix = np.zeros((len(x), (poly_degree + 1)**2 - 1) ) #creating empty feature matrix

k = 0
for i in range(poly_degree + 1):     #nested loop, to get all combinations of x and y
    for j in range(poly_degree + 1): 
        if i != 0 or j !=0:                    
            feature_matrix[:,k] = x**i * y**j
            k += 1

X_train, X_test, z_train, z_test = train_test_split(feature_matrix, z) #splitting my data into test and train data

beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train #calculating beta

print(beta_ols)




"""

for i in range(poly_degree): 
    Xy_nested[i] = x**(i+1) * y #filling my empty arrays with elements of (x)^i times y
    xY_nested[i] = x * y**(i+1) #filling my empty arrays with elements of x times (y)^i
        
for i in range( poly_degree ): #filling my (xy)^i list. Start range is 1, as i do not want to overlap with first elements of the Xy and xY lists
    xy_nested[i] = (x * y) ** (i+1)
        

#print(len(Xy_nested))
#print(len(xY_nested))


#x_coloumn = x.reshape(-1, 1) #All coloumns are arrays
#y_coloumn = y.reshape(-1, 1)
#xy_coloumn = xy.reshape(-1,1)

arrays_to_stack = []
arrays_to_stack.append(x )
arrays_to_stack.append(y )
#arrays_to_stack.append(x - np.mean(x))
#arrays_to_stack.append(y - np.mean(y))

for i,j in zip(range(poly_degree), range(poly_degree - 1)):
    arrays_to_stack.append(xy_nested[j] )
    arrays_to_stack.append(Xy_nested[i] )
    arrays_to_stack.append(xY_nested[i] )
    #arrays_to_stack.append(xy_nested[i] - np.mean(xy_nested[i]))
    #arrays_to_stack.append(Xy_nested[i] - np.mean(Xy_nested[i]))
    #arrays_to_stack.append(xY_nested[i] - np.mean(xY_nested[i]))

    
#print("lenght of arrays_to_stack:",len(arrays_to_stack))



feature_matrix = np.stack((arrays_to_stack),axis = 1)

#print("feature_matrix shape = ",feature_matrix.shape)

print(feature_matrix)


X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y)

#print("X_train shape = ",X_train.shape)

A = X_train.T @ X_train

#non_zero_rows_mask = np.any(A != 0, axis=1)

# Use the mask to extract rows that are not all zeros
#filtered_A = A[non_zero_rows_mask]

#tranpose_A = filtered_A.T

#test = tranpose_A @ filtered_A

#inverse = np.linalg.inv(test)

#B = np.linalg.inv(A)

print(B)



print("Filtered_A:", filtered_A)
print("A:", A)
print(filtered_A.shape)
print(test.shape)

#print("A:",A)
#print("X_train",X_train)
#print("B_matrix=",B)

sys.exit()

beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

print(beta_ols)




#we rescale our feature matrix by subtracting the mean value of each coloumn to each coloumn-element


# We split the data in test and training data. We specfify X
X_train, X_test, y_train, y_test = train_test_split(X, y)


#This beta is the beta used for predicting the output of the trainingset. It is from the traingset we get beta. 

#beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y

print(beta_ols)
#predicted output values from the training set. This is y_tilde, which is the predicted outcome of our model 
y_predict_train = X_train @ beta_ols
y_predict_test = X_test @ beta_ols

#MSE between trained and predicted outcomes y
MSE_train = mean_squared_error(y_train,y_predict_train)

print("MSE_train_ols", MSE_train)

#MSE between test and predicted outcomes y
MSE_test = mean_squared_error(y_test,y_predict_test)

print("MSE_test_ols", MSE_test)

"""





