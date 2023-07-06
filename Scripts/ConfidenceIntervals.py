#Confidence Intervals
import numpy as np
from scipy.stats import norm
from scipy.stats import t

x = [1, 2, 3, 4, 5, 6, 7]

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


confidence_intervals(x, 0.95)