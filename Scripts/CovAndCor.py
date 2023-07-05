#Covariance

import numpy as np

series_x = [1, 3, 3, 7, 10]
series_y = [8, 4, 2, 2, 1]

#Calculate the means of the series
mean_x = sum(series_x) / len(series_x)
mean_y = sum(series_y) / len(series_y)

#Calculate the differences from the means using a list comprehension
diff_x = [x - mean_x for x in series_x]
diff_y = [y - mean_y for y in series_y]

#Multiply the differences
diffXdiff = np.multiply(diff_x,diff_y)

#Sum the products
sum_diffXdiff = np.sum(diffXdiff)

#Divide by N or n-1
covariance = sum_diffXdiff / (len(series_x) - 1)

print(f"The covariance of the two series is {covariance}.")

#As a function
def calculate_covariance(series_x, series_y):
    mean_x = sum(series_x) / len(series_x)
    mean_y = sum(series_y) / len(series_y)
    diff_x = [x - mean_x for x in series_x]
    diff_y = [y - mean_y for y in series_y]
    diffXdiff = np.multiply(diff_x,diff_y)
    sum_diffXdiff = np.sum(diffXdiff)
    covariance = sum_diffXdiff / (len(series_x) - 1)
    return covariance

calculate_covariance(series_x, series_y)

#The easy way...
series_x = [1, 3, 3, 7, 10]
series_y = [8, 4, 2, 2, 1]

covariance_xy = np.cov(series_x, series_y)[0][1]
print(covariance_xy)


#Correlation

series_x = [1, 3, 3, 7, 10]
series_y = [8, 4, 2, 2, 1]

#Calculate the means of the series
mean_x = sum(series_x) / len(series_x)
mean_y = sum(series_y) / len(series_y)

#Calculate the differences from the means using a list comprehension
diff_x = [x - mean_x for x in series_x]
diff_y = [y - mean_y for y in series_y]

#Multiply the differences
diffXdiff = np.multiply(diff_x,diff_y)

#Sum the products
sum_diffXdiff = np.sum(diffXdiff)

#Square the differences
diff_x_squared = [x ** 2 for x in diff_x]
diff_y_squared = [y ** 2 for y in diff_y]

#Sum the squared differences
sum_diff_x_squared = np.sum(diff_x_squared)
sum_diff_y_squared = np.sum(diff_y_squared)

#Multiply the sums
sdxsXsdys = sum_diff_x_squared * sum_diff_y_squared

#Take the square root of the multiplied sums
sqrt_sdxsXsdys = np.sqrt(sdxsXsdys)

#Finally, compute the correlation coefficent
correlation = sum_diffXdiff / sqrt_sdxsXsdys

print(f'The data have a correlation coefficient of {correlation}')

#As a function
def calculate_correlation(series_x, series_y):
    mean_x = sum(series_x) / len(series_x)
    mean_y = sum(series_y) / len(series_y)
    diff_x = [x - mean_x for x in series_x]
    diff_y = [y - mean_y for y in series_y]
    diffXdiff = np.multiply(diff_x,diff_y)
    sum_diffXdiff = np.sum(diffXdiff)
    diff_x_squared = [x ** 2 for x in diff_x]
    diff_y_squared = [y ** 2 for y in diff_y]
    sum_diff_x_squared = np.sum(diff_x_squared)
    sum_diff_y_squared = np.sum(diff_y_squared)
    sdxsXsdys = sum_diff_x_squared * sum_diff_y_squared
    sqrt_sdxsXsdys = np.sqrt(sdxsXsdys)
    correlation = sum_diffXdiff / sqrt_sdxsXsdys
    return correlation

calculate_correlation(series_x,series_y)

#The easy way...
series_x = [1, 3, 3, 7, 10]
series_y = [8, 4, 2, 2, 1]

correlation_xy = np.corrcoef(series_x, series_y)[0][1]
print(correlation_xy)