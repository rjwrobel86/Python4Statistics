import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

#Make/fake some data
np.random.seed(1663)
n = 1000

salaries = np.random.normal(50000, 10000, n)
majors = ['Engineering', 'Computer Science', 'Sociology', 'Education'] 
school_types = ['Public', 'Private']
ages = np.random.randint(22, 65, n)
genders = ['Male', 'Female']

data = {
    'Major': np.random.choice(majors, size=n),
    'Salary': salaries,
    'School_Type': np.random.choice(school_types, size=n),
    'Age': ages,
    'Gender': np.random.choice(genders, size=n)
}

df = pd.DataFrame(data)

#Box plots by factor (major)
major_grouped_salary = df.groupby('Major')['Salary'].apply(list)
categories = major_grouped_salary.index
values = major_grouped_salary.values
plt.boxplot(values, labels=categories)
plt.xlabel('Major')
plt.ylabel('Salary')
plt.show()

#Means by factor (major)
grouped_means = df.groupby('Major').mean()
print(grouped_means)

#ANOVA Assumption 1: Normally distributed data
grouped_data = df.groupby('Major')

#Plot histogram for each group
for group_name, group_data in grouped_data:
    plt.hist(group_data['Salary'], bins='auto', alpha=0.7, label=group_name)

plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Histogram of Salary by Major')
plt.legend()

plt.show()

#ANOVA Assumption 1: Normally distributed data
grouped_data = df.groupby('Major')

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

#Flatten the subplots grid
axs = axs.flatten()

#Plot histogram for each group in a separate subplot
for i, (group_name, group_data) in enumerate(grouped_data):
    axs[i].hist(group_data['Salary'], bins='auto', alpha=0.7)
    axs[i].set_title(group_name)
    axs[i].set_xlabel('Salary')
    axs[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

#Shapiro-Wilk statistical test for normality assumption

grouped_data = df.groupby('Major')

#Perform Shapiro-Wilk test for each group
for group_name, group_data in grouped_data:
    salaries = group_data['Salary']
    shapiro_test = shapiro(salaries)
    test_statistic = shapiro_test.statistic
    p_value = shapiro_test.pvalue
    
    print(f"Shapiro-Wilk Test Results for {group_name}")
    print("Test Statistic:", test_statistic)
    print("p-value:", p_value)
    
#Null hypothesis: Data are normally distributed. If p is less than cutoff, reject the null of normally distributed data.

#Levene statistical test for homogeneous variance
levene_test = levene(df['Salary'][df['Major'] == 'Computer Science'],
                     df['Salary'][df['Major'] == 'Engineering'],
                     df['Salary'][df['Major'] == 'Education'],
                     df['Salary'][df['Major'] == 'Sociology'])

test_statistic = levene_test.statistic
p_value = levene_test.pvalue

print("Levene's Test Results:")
print("Test Statistic:", test_statistic)
print("p-value:", p_value)
#Null hypothesis: Variances are equal.  If p-value is less than cutoff, reject null that variances are equal

#Perform one-way ANOVA with statsmodels
model = ols('Salary ~ Major', data=df).fit()
anova_table = sm.stats.anova_lm(model)
print(anova_table)

#Alternatively, perform one-way ANOVA with scipy's f_oneway function
groups = []
for major, group in df.groupby('Major')['Salary']:
    groups.append(group)

f_stat, p_value = f_oneway(*groups)

print("F-statistic:", f_stat)
print("p-value:", p_value)

#The associated p-value indicates the probability of observing such an F-statistic under the null hypothesis (no group differences).
#If the p-value is below cutoff, we reject the null hypothesis and conclude that there are significant differences between the groups.

#Create some differences!
df.loc[df['Major'] == 'Engineering', 'Salary'] *= 1.25
df.loc[df['Major'] == 'Education', 'Salary'] *= 0.75
df.loc[df['Major'] == 'Sociology', 'Salary'] -= 3000

major_grouped_salary = df.groupby('Major')['Salary'].apply(list)
categories = major_grouped_salary.index
values = major_grouped_salary.values
plt.boxplot(values, labels=categories)
plt.xlabel('Major')
plt.ylabel('Salary')
plt.show()

grouped_means = df.groupby('Major').mean()
print(grouped_means)

#Perform one-way ANOVA with statsmodels
model = ols('Salary ~ Major', data=df).fit()
anova_table = sm.stats.anova_lm(model)

print(anova_table)

#Perform one-way ANOVA with scipy's f_oneway function
groups = []
for major, group in df.groupby('Major')['Salary']:
    groups.append(group)

f_stat, p_value = f_oneway(*groups)

#Print the results
print("F-statistic:", f_stat)
print("p-value:", p_value)

#Two Way ANOVA
#Reset fake  data
np.random.seed(1663)
n = 1000

salaries = np.random.normal(50000, 10000, n)
majors = ['Engineering', 'Computer Science', 'Sociology', 'Education'] 
school_types = ['Public', 'Private']
ages = np.random.randint(22, 65, n)
genders = ['Male', 'Female']

data = {
    'Major': np.random.choice(majors, size=n),
    'Salary': salaries,
    'School_Type': np.random.choice(school_types, size=n),
    'Age': ages,
    'Gender': np.random.choice(genders, size=n)
}

df = pd.DataFrame(data)

#Means by Major
grouped_means = df.groupby('Major').mean()
print(grouped_means)

#Means by School Type
grouped_means = df.groupby('School_Type').mean()
print(grouped_means)

#Perform two-way ANOVA
formula = 'Salary ~ C(Major) + C(School_Type) + C(Major):C(Gender)'
model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

#Create some differences and test again
df.loc[df['Major'] == 'Engineering', 'Salary'] *= 1.25
df.loc[df['Major'] == 'Education', 'Salary'] *= 0.75
df.loc[df['Major'] == 'Sociology', 'Salary'] -= 3000
df.loc[df['School_Type'] == 'Private', 'Salary'] += 6000

#Means by Major
grouped_means = df.groupby('Major').mean()
print(grouped_means)

#Means by School Type
grouped_means = df.groupby('School_Type').mean()
print(grouped_means)

formula = 'Salary ~ C(Major) + C(School_Type) + C(Major):C(School_Type)'
model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)




