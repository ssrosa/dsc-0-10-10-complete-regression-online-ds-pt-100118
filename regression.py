import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import math

#take in an iterable, calculate the mean and subtract the mean value
# from each element , creating and returning a new list. 
def mean_normalize(var):
    var_arr = np.array(var)
    var_normed = var_arr - var_arr.mean()
    return list(var_normed)

#calculate the dot product of two iterables 
def dot_product(x,y):
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    xy = x_arr * y_arr
    
    return xy.sum()

def covariance(var1, var2):
    var1_normed = mean_normalize(var1)
    var2_normed = mean_normalize(var2)
    
    n = len(var1_normed)
    
    return dot_product(var1_normed, var2_normed) / (n - 1)


def correlation(var1,var2):
    
    var1_normed = mean_normalize(var1)
    var2_normed = mean_normalize(var2)
    
    numerator = dot_product(var1_normed, var2_normed)
    
    denominator = math.sqrt(dot_product(var1_normed, var1_normed) * dot_product(var2_normed, var2_normed))
    
    return numerator / denominator

# Write the function to calculate slope as: 
# (mean(x) * mean(y) – mean(x*y)) / ( mean (x)^2 – mean( x^2))
def calc_slope(xs,ys):
    
    slope = ((xs.mean() * ys.mean()) - (xs*ys).mean()) / ((xs.mean()**2) - (xs**2).mean()) 
    
    return slope

# use the slope function with intercept formula to return calculate slop and intercept from data points
def best_fit(xs,ys):
    m = calc_slope(xs, ys)
    b = ys.mean() - (m * xs.mean())

    return m, b

#takes in slope, intercept and X vector and calculates the regression line using Y= mX+c for each point in X.
def reg_line (m, b, xs):
    Y = (xs * m) + b
    
    return Y

# Calculate sum of squared errors between regression and mean line 
def sq_err(ys_a, ys_b):
	diffs = ys_a - ys_b
	diffs_sqd = diffs ** 2
	squared_error = diffs_sqd.sum()

	return squared_error


# Calculate Y_mean , squared error for regression and mean line , and calculate r-squared
def r_squared(ys_real, ys_predicted):
    y_mean = ys_real.mean()
    
    sse = sq_err(ys_real, ys_predicted)  #real - predicted
    
    sst = sq_err(ys_real, y_mean)  # real - mean
    
    r_squared = 1 - (sse / sst)
    
    return r_squared

