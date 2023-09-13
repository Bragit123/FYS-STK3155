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
poly_degree = 5

# Make data set. meshgrid is only for plotting 
#x = np.arange(2, 3, 0.01)
#y = np.arange(2, 3, 0.01)

x = np.random.rand(1000) #need random numbers between 0 and 1, or else you get singular matrix
y = np.random.rand(1000)

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

    lambdaa = [0.0001, 0.001, 0.01, 0.1, 1 ] 
    X_train_ridge, X_test_ridge, z_train_ridge, z_test_ridge = train_test_split(feature_matrix_list[i], z) #splitting my data into test and train data for Ridge regression
    Identity.append(np.eye((i + 2)**2 - 1))
    beta_ridge = np.linalg.inv(X_train_ridge.T @ X_train_ridge + lambdaa[i] * Identity[i]) @ X_train_ridge.T @ z_train_ridge
    
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

fig3.savefig('MSE_and_R2_vs_Degree_LASSO')

#fig2.savefig('MSE_and_R2_vs_Degree_RIDGE')


#plt.savefig('MSE_and_R2_vs_Degree_OLS')


plt.show()
"""

#PART B ADDING RIDGE REGRESSION

lambdaa = [0.0001, 0.001, 0.01, 0.1, 1 ]
Identity = np.eye(6)
Identity2 = np.eye(11)
Identity3 = np.eye(16)
MSE_test_ridge = []
MSE_train_ridge = []
difference = []

MSE_ridge_10_test = []
MSE_ridge_10_train = []
MSE_ridge_15_test = []
MSE_ridge_15_train = []

for i in lambdaa:
    beta_ridge = np.linalg.inv(X_new_train.T @ X_new_train + i * Identity) @ X_new_train.T @ y_train
    beta_ridge_10 = np.linalg.inv(X_10_train.T @ X_10_train + i * Identity2) @ X_10_train.T @ y_train
    beta_ridge_15 = np.linalg.inv(X_15_train.T @ X_15_train + i * Identity3) @ X_15_train.T @ y_train

    y_predicted_test_ridge = X_new_test @ beta_ridge
    y_predicted_train_ridge = X_new_train @ beta_ridge

    y_predicted10_test_ridge = X_10_test @ beta_ridge_10
    y_predicted10_train_ridge = X_10_train @ beta_ridge_10

    y_predicted15_test_ridge = X_15_test @ beta_ridge_15
    y_predicted15_train_ridge = X_15_train @ beta_ridge_15

    MSE_test_ridge.append(mean_squared_error(y_test,y_predicted_test_ridge))
    MSE_train_ridge.append(mean_squared_error(y_train,y_predicted_train_ridge))

    MSE_ridge_10_test.append(mean_squared_error(y_test,y_predicted10_test_ridge))
    MSE_ridge_10_train.append(mean_squared_error(y_train,y_predicted10_train_ridge))

    MSE_ridge_15_test.append(mean_squared_error(y_test,y_predicted15_test_ridge))
    MSE_ridge_15_train.append(mean_squared_error(y_train,y_predicted15_train_ridge))







    #X_train_list.append(X_train)
    #X_test_list.append(X_test)
    #z_train_list.append(z_train)
    #z_test_list.append(z_train)
    
    #print(f"inside for loop:",X_train)


#print(X_train_list[1])

#print(f"outside for loop:",X_train)



    #columns_to_remove = [0, -i]

    # Create a new matrix with columns removed
    #new_matrix = np.delete(matrix, columns_to_remove, axis=1)

    feature_matrix_list.append(feature_matrix.copy())
    #print(f"Feature matrix number {i} is : {feature_matrix_list[i]}")

#print(feature_matrix_list[0])
#print(feature_matrix_list[1])

print("feature matrix:",feature_matrix)



non_zero_feature_matrix_list = []


for i in range(len(feature_matrix_list)):
    raveled_matrix = feature_matrix_list[i].ravel()
    non_zero_elements = raveled_matrix[raveled_matrix != 0]
    new_shape = feature_matrix_list[i].shape
    reshaped_matrix = non_zero_elements.reshape(new_shape)

    non_zero_feature_matrix_list.append(reshaped_matrix)




    #non_zero_feature_matrix_list.append(reshaped_matrix)


#for i in range(len(feature_matrix_list)):
#    print(f"Shape of poly degree{i}: {non_zero_feature_matrix_list[i].shape}")


print(non_zero_feature_matrix_list[0])
print(feature_matrix_list[0])
    

filtered_A = []
for i in feature_matrix_list:
    non_zero_rows_mask = np.any(i != 0, axis=1)
    #filtered_A = A[non_zero_rows_mask]
    non_zero_feature_matrix = i[non_zero_rows_mask]
    #print(non_zero_feature_matrix.shape)
    #filtered_A.append(non_zero_feature_matrix)

#print(filtered_A)

# Use the mask to extract rows that are not all zeros
#filtered_A = A[non_zero_rows_mask]

X_train, X_test, z_train, z_test = train_test_split(feature_matrix, z) #splitting my data into test and train data

beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train #calculating beta


#predicted output values from the training set. This is Z_tilde, which is the predicted outcome of our model 
z_predict_train = X_train @ beta_ols
z_predict_test = X_test @ beta_ols

MSE_ols_train = mean_squared_error(z_train,z_predict_train)
MSE_ols_test = mean_squared_error(z_test,z_predict_test)

#R2_ols_train = r2_score(z_train,z_predict_train)
#R2_ols_test = r2_score(z_test,z_predict_test)





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





