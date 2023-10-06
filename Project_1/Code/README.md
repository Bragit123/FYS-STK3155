# Code for project 1
This folder contains the code for the project. These are all python codes that
are run simply by using 

$ python3 <filename>

The only exception to this is the functions.py.

## The codes

### functions.py
Contains functions that we use throughout the other codes, and is imported as a
module in most other codes in this repository. Since functions.py only contains
functions, running it will not produce any output.

### franke_visualize.py
Creates a 3D plot showing what the Franke function looks like.

### ols.py, ridge.py and lasso.py
ols.py, ridge.py and lasso.py produce a random set of points $(x,y)$, and
compute the Franke function corresponding to these points.
They then compute the ordinary least squares (OLS), Ridge and Lasso fit
respectively, and finds the mean square error (MSE) and $R^2$-value for both the
training and test data, and plots these as functions of degrees (for OLS) or
lambda (for Ridge and Lasso).

### bootstrap.py
This function works in the same way as ols.py, except it uses bootstrap
resampling. It plots the mean square error (MSE) of the training and test data
as a function of polynomial degree.

### crossvalidate.py and crossvalidatelasso.py
These codes do the same as ols.py, ridge.py and lasso.py, but they also compute
the mean square error (MSE) using cross validation. The codes returns a plot of
the MSE as a function of polynomial degree (for OLS) or lambda (for Ridge and Lasso) for the test data with and without
crossvalidation for the OLS and Ridge methods (crossvalidate.py) and the Lasso
method (crossvalidatelasso.py).

### terrain.py and terrainlasso.py
These codes compute the OLS and Ridge (terrain.py) and Lasso (terrainlasso.py)
fits to the terrain data given in SRTM_data_Norway_1.tif. They compute the fits
both with and without cross validation, and produces contourplots of the
predicted terrain, and the mean square error for the fits as functions of the
polynomial degree (for OLS) or lambda (for Ridge and Lasso).