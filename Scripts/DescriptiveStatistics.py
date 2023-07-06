#Descriptive Statistics (the hard way!?)

#Function to calculate mean
def calculate_mean(series_x):
    mean = sum(series_x)/len(series_x)
    return mean

#Function to calculate median
def calculate_median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 1:
        return int(sorted_values[n//2]) #int gets rid of floating point decimal
    else:
        return int((sorted_values[n//2 - 1] + sorted_values[n//2]) / 2)

even = [1, 2, 3, 8, 3, 4, 1, 6]
calculate_median(even)

odd = [1, 4, 5, 6, 8]
calculate_median(odd)

#Function to calculate mode
def calculate_mode(values):
    value_counts = {}
    for value in values:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    #print(value_counts)
    #print(f"The most commonly occuring number is {max(value_counts)}")
    #higest_count = max(value_counts.values())
    #print(f"It occurs {higest_count} times")
    return max(value_counts)

x = [1, 2, 3, 3, 4, 5, 6]
calculate_mode(x)

#Function to calculate range
def calculate_range(values):
    return max(values) - min(values)

x = [1, 2, 3, 3, 4, 5, 6]
calculate_range(x)

#Function to calculate variance
def calculate_variance(values):
    mean = calculate_mean(values)
    return sum((x - mean) ** 2 for x in values) / len(values)

x = [1, 2, 3, 3, 4, 5, 6]
calculate_variance(x)

#Import Numpy so we can use the standard deviation funciton
import numpy as np

#Function to calculate standard deviation
def calculate_stdev(values):
    return np.sqrt(calculate_variance(values))

calculate_stdev(x)

def calculate_covariance(series_x, series_y):
    mean_x = sum(series_x) / len(series_x)
    mean_y = sum(series_y) / len(series_y)
    diff_x = [x - mean_x for x in series_x]
    diff_y = [y - mean_y for y in series_y]
    diffXdiff = np.multiply(diff_x,diff_y)
    sum_diffXdiff = np.sum(diffXdiff)
    covariance = sum_diffXdiff / (len(series_x) - 1)
    return covariance

series_x = [1, 3, 3, 7, 10]
series_y = [8, 4, 2, 2, 1]

calculate_covariance(series_x, series_y)

#Function to calculate correlation
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

series_x = [1, 3, 3, 7, 10]
series_y = [8, 4, 2, 2, 1]

calculate_correlation(series_x, series_y)


from scipy.stats import norm
from scipy.stats import t

series_x = [1, 2, 3, 4, 5, 6, 7]

#Function to calculate confidence intervals
def confidence_intervals(series, level):
    mean = np.mean(series)
    std = np.std(series)
    n = len(series)
    df = n - 1
    se = std / np.sqrt(n)
    z_cv= norm.ppf(level)
    t_cv = t.ppf(1 - level / 2, df=df)
    pctconfident = str(int(level * 100))
    cl = pctconfident + "% confidence level"
    if len(series) >= 30:
        cv = z_cv
        me = cv * se
        test = 'Z Test '
        letter = 'T '
    else: 
        cv = t_cv
        me = cv * se
        test = 'T Test '
        letter = 'T '


    print(f'The sample mean is {mean}.')
    print(f'The test used is a {test}and the critical value of {letter}is {cv}.')
    print(f'The sample mean has a margin of error of: {me}.')
    print(f'The sample mean has an upper confidence interval of: {mean + me}')
    print(f'The sample mean has a lower confidence interval of {mean - me}')
    print(f'At the {cl}, the actual populaiton mean will be within {mean - me} and {mean + me}, {pctconfident} out of 100 times.')


confidence_intervals(series_x ,0.95)

#Import Panadas so we can import and manipulate data
import pandas as pd
df = pd.read_csv("monthly_sales.csv")

#Define the columns we are interested in (used lower in the code)
columns = ['Sales', 'Cost', 'Revenue']

for column in columns:
    values = df[column].tolist()
    print(f"Descriptive Statistics for {column}:")
    print(f"Mean: {calculate_mean(values)}")
    print(f"Median: {calculate_median(values)}")
    print(f"Mode: {calculate_mode(values)}")
    print(f"Range: {calculate_range(values)}")
    print(f"Variance: {calculate_variance(values)}")
    print(f"Standard Deviation: {calculate_stdev(values)}")
    print("\n")

#Pandas can, of course, do all of what we've done so far much easier
print(df['Sales'].mean())
print(df['Sales'].median())
print(df['Sales'].mode()[0])
print(df['Sales'].max())
print(df['Sales'].min())
print(df['Sales'].var())
print(df['Sales'].std())

#Even easier, though not better!
df.describe()
