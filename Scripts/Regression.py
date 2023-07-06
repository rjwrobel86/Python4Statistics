import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, het_breuschpagan, linear_reset  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

import scipy.stats as stats
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('regression_data.csv')
df.head()

#Create scatter plots of all x variables against y, w/o trendline
def create_scatter_plots(dataframe, y_variable):
    x_variables = dataframe.columns.tolist()
    for x_variable in x_variables:
        if x_variable != y_variable:
            plt.scatter(dataframe[x_variable], dataframe[y_variable])
            plt.xlabel(x_variable)
            plt.ylabel(y_variable)
            plt.title(f"Scatter Plot: {x_variable} vs {y_variable}")
            plt.show()  
            
            
create_scatter_plots(df,'y')

#Create scatter plots of all x variables against y, w/ trendline
#Categorical variable breaks this, so it is was removed. New code checks if categorical and removes any variables that will cause a problem
def create_scatter_plots(dataframe, y_variable):
    #x_variables.remove('cat')
    x_variables = dataframe.columns.tolist()
    categorical_variables = []

    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            categorical_variables.append(column)

    x_variables = [x for x in x_variables if x not in categorical_variables]


    for x_variable in x_variables:
        if x_variable != y_variable:
            x = dataframe[x_variable]
            y = dataframe[y_variable]

            slope, intercept = np.polyfit(x, y, 1)
            trend_line = slope * x + intercept

            plt.scatter(x, y)
            plt.plot(x, trend_line, color='red', label='Trend Line')
            plt.xlabel(x_variable)
            plt.ylabel(y_variable)
            plt.title(f"Scatter Plot: {x_variable} vs {y_variable}")
            plt.legend()
            plt.show()
            
create_scatter_plots(df,'y')


def create_scatter_plots_with_seaborn(dataframe, y_variable):
    #x_variables.remove('cat')
    x_variables = dataframe.columns.tolist()
    categorical_variables = []

    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            categorical_variables.append(column)

    x_variables = [x for x in x_variables if x not in categorical_variables]

    for x in x_variables:
        sns.lmplot(x=x,y='y', fit_reg=True, data=df)

create_scatter_plots_with_seaborn(df,y)


#Plot everything against everything with Seaborn
sns.pairplot(df)

#Transforming Variables - Logs, Squares, and Inverses
df['log_x'] = np.log(df['x']) #Log(x)
df['x_squared'] = df['x'] * df['x'] #x^2
df['x_inverse'] = 1/df['x'] #1/x

#Transforming Variables - Interaction Terms
df['xINTERACTd1'] = df['x'] * df['d1'] #Categorical
df['xINTERACTz'] = df['x'] * df['z'] #Continuous

#Transforming Variables - Lagged X Variables
df['x_lag_1'] = df['x'].shift(1)  #1 Period Lag
df['x_lag_2'] = df['x'].shift(2)  #2 Period Lag 
df['x_lag_3'] = df['x'].shift(3)  #3 Period Lag 

#Lagged Y Variable
df['y_lag_1'] = df['y'].shift(1)  #1 Period Lag 
df['y_lag_2'] = df['y'].shift(2)  #2 Period Lag 
df['y_lag_3'] = df['y'].shift(3)  #3 Period Lag 

df.head()
df.describe()

#Lagged variables create unequal columns
#Remove unequal columns / missing rows
df.dropna(inplace=True)
df.head()
df.describe()

#Regression
model = sm.OLS.from_formula('y ~ x + z + x2 + y2', df)
results = model.fit()
results.summary()

#Alternative formula
X = df[['x','z','x2','y2']]
y = df['y'] 
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
results.summary()

#Gather Regression Statistics and Values
rsquared = results.rsquared
adj_rsquared = results.rsquared_adj
fstat = results.fvalue
aic = results.aic
bic = results.bic

parameters = results.params
tvalues = results.tvalues
pvalues = results.pvalues

ci = results.conf_int(alpha=0.05)

yhat = results.fittedvalues
e = results.resid

#Add fitted values and residuals to original dataframe - NOT TYPICAL
df['yhat'] = yhat
df['e'] = e
df.head()

#Plot Y vs fitted values (yhat)
sns.lmplot(x='y',y='yhat', fit_reg=True, data=df)

#Plot fitted values (yhat) vs residuals (e)
sns.lmplot(x='e',y='yhat', fit_reg=True, data=df)

#Plot residuals against x values
x_vars =['x','x2','y2','z','d1']
for i in x_vars:
    sns.lmplot(x=i,y='e', fit_reg=True, data=df)    

sm.graphics.plot_partregress_grid(results)

#Correlation matrix
df.corr() 

cor = df.corr()
sns.heatmap(cor, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

#Set plot title
plt.title('Correlation Matrix Heatmap')

#Display the heatmap
plt.show()


#Multicollinearity test
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF) for Multicollinearity:")
print(vif)
# Interpretation:
#VIF values greater than 1 indicate the presence of multicollinearity.
#A commonly used threshold is VIF > 5 or VIF > 10 to identify problematic multicollinearity.
#Higher VIF values suggest stronger correlation among the predictor variables.
#Variables with high VIF may need further examination or potential remedial actions such as variable selection or transformation.


#Variables with VIF greater than 1, 5, and 10
vif_greater_than_1 = vif[vif['VIF'] > 1]['Variable'].tolist()
vif_greater_than_5 = vif[vif['VIF'] > 5]['Variable'].tolist()
vif_greater_than_10 = vif[vif['VIF'] > 10]['Variable'].tolist()

print("\nVariables with VIF > 1:", vif_greater_than_1)
print("Variables with VIF > 5:", vif_greater_than_5)
print("Variables with VIF > 10:", vif_greater_than_10)

#Histogram to check for normally distributed residuals
plt.hist(e)

#Fitted values vs residuals to check for nonconstant error variance
plt.scatter(yhat, e)
plt.show()

#Heteroskedasticity tests
white_test = het_white(results.resid, X)
bp_test = het_breuschpagan(results.resid, X)
print("Heteroskedasticity Tests:")
print("White's Test:")
print("LM Statistic:", white_test[0])
print("LM P-value:", white_test[1])
print("F-Statistic:", white_test[2])
print("F P-value:", white_test[3])
#Interpretation:
#If the p-value of White's test is above the significance level (e.g., 0.05), we fail to reject the null hypothesis of homoskedasticity.
#If the p-value of Breusch-Pagan test is above the significance level (e.g., 0.05), we fail to reject the null hypothesis of homoskedasticity.


#Categorical interaction term
model = sm.OLS.from_formula('y ~ x + d1 + x:d1', df)
results = model.fit()
r = model.fit()
results.summary()

#Plot interaction terms at set levels of d1
sns.catplot(x='d1', y='interaction_term', hue='categorical_var2', kind='bar', data=df)

#Continuous interaction term
model = sm.OLS.from_formula('y ~ x + z + zx', df)
results = model.fit()
r = model.fit()
results.summary()

#Plot inteaction terms at set levels of Z


predictions = pd.DataFrame({'x': [1, 2, 2, 4, 5],
                       'z': [1, 1, 4, 3, 3],
                       'cat':['a','a','a','d','d'],
                        'x2':[100,100,100,100,100]})

predictions['yhat'] = results.predict(predictions)
print(predictions)

#sns.lmplot(x='x',y='y', fit_reg=False, data=df)

print(r.params)
plt.scatter(df['x'], df['y'])
plt.plot(df['x'], r.params[0] + r.params[1]*df['x'] + r.params[2] * 0 + r.params[3] * 0, color='red')
plt.plot(df['x'], r.params[0] + r.params[1]*df['x'] + r.params[2] * 100 + r.params[3] * 100, color='purple')
plt.plot(df['x'], r.params[0] + r.params[1]*df['x'] + r.params[2] * 200 + r.params[3] * 200, color='blue')


#Comparing Models with F-Tests
#More complex model is model 2.  F-Test checks to see if coefficients for extra variables are zero-ish
model1 = sm.OLS.from_formula('y ~ x + z', df).fit()
model2 = sm.OLS.from_formula('y ~ x + z + x2 + y2', df).fit()
from statsmodels.stats.anova import anova_lm

anova_result = anova_lm(model1, model2)

p_values = anova_result.iloc[:, -1][1]
print(p_values)

#print(p_values)
#Since P-Value is below cutoff, 0.05 for our purposes, we reject the null hypothesis that the extra variables have zero predictive power

#Createing an F-Test function
def f_tester(model1,model2,p_cutoff):
    anova_results = anova_lm(model1, model2)
    p_value = anova_result.iloc[:, -1][1]
    if p_value < p_cutoff:
        print(f'Your test statistic p-value is {p_value}, \nwhich is less than the cutoff value of {p_cutoff}, so reject \nthe null that the extra variables have no predictive value.')
    else:
        print(f'Your test statistic p-value is {p_value}, which is greater than the cutoff value of {p_cutoff}, so do not reject the null that the extra variables have no predictive value.')
    print(f'\n Reject the Null: {p_value < p_cutoff}')
f_tester(model1,model2,0.05)

#Jarque-Bera test for normality of residuals:

def jb(significance_level):
    jb_test = jarque_bera(results.resid)
    #print("Jarque-Bera Test for Normality of Residuals:")
    if jb_test[1] > significance_level:
        print(f'\nThe P-Value is {jb_test[1]} which is greater than the significance level of {significance_level}.')
        print("\nDo not reject the null hypothesis that the residuals are normally distributed.")
    else:
        print(f'\nThe P-Value is {jb_test[1]} which is greater than the significance level of {significance_level}.')
        print("\nReject the null hypothesis that the residuals are normally distributed.")

jb(0.05)

#QQ Plot for normality of residuals
model = sm.OLS.from_formula('y ~ x', df)
results = model.fit()

sm.qqplot(results.resid, line='s', dist=stats.norm)
plt.title("QQ Plot of Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()

#Perform the Ramsey Reset test
reset_result = linear_reset(results)
print(reset_result)
#Print the test statistics and p-value
print("Ramsey Reset test f-value:", reset_result.statistic)
print("Ramsey Reset test p-value:", reset_result.pvalue)

def reset_test(model,p):
    results = model
    reset_result = linear_reset(results)
    if reset_result.pvalue < p:
        print(f'The RESET test p-value is {reset_result.pvalue} which is less than the critical value of {p}.')
        print('Reject the null that the model is correctly specified and consider non-linear relationships.')
    else:
        print(f'The RESET test p-value is {reset_result.pvalue} which is greater than the critical value of {p}.')
        print('Do not reject the null that the model is correclt specified.')

reset_test(model1,0.05)

#Comparing models with other measures
models = ['model1','model2']
r_squared = [model1.rsquared, model2.rsquared]
adj_r_squared = [model1.rsquared_adj, model2.rsquared_adj]
aic = [model1.aic, model2.aic]
bic = [model1.bic, model2.bic]

data = {
    'Model': models,
    'R-squared': r_squared,
    'Adjusted R-squared': adj_r_squared,
    'AIC': aic,
    'BIC': bic
}
df_stats = pd.DataFrame(data)

print(df_stats)

for_model1 = 0
for_model2 = 0

# Comparison for R-squared
if r_squared[0] > r_squared[1]:
    print("Model 1 has a higher R-squared.")
    for_model1 += 1
elif r_squared[1] > r_squared[0]:
    print("Model 2 has a higher R-squared.")
    for_model2 += 1

else:
    print("R-squared values are equal for both models.")

# Comparison for adjusted R-squared
if adj_r_squared[0] > adj_r_squared[1]:
    print("Model 1 has a higher adjusted R-squared.")
    for_model1 += 1
elif adj_r_squared[1] > adj_r_squared[0]:
    print("Model 2 has a higher adjusted R-squared.")
    for_model2 += 1
else:
    print("Adjusted R-squared values are equal for both models.")

# Comparison for AIC
if aic[0] < aic[1]:
    print("Model 1 has a lower AIC.")
    for_model1 += 1
elif aic[1] < aic[0]:
    print("Model 2 has a lower AIC.")
    for_model2 += 1

else:
    print("AIC values are equal for both models.")

# Comparison for BIC
if bic[0] < bic[1]:
    print("Model 1 has a lower BIC.")
    for_model1 += 1
elif bic[1] < bic[0]:
    print("Model 2 has a lower BIC.")
    for_model2 += 1

else:
    print("BIC values are equal for both models.")
    
if for_model1 > for_model2:
    print('Based on all for selection criteria, Model1 is a better fit')
elif for_model1 < for_model2:
    print('Based on all for selection criteria, Model2 is a better fit')
else:
    print('Based on all for selection criteria, Model1 and Model2 fit the data equally well')


#Comparing Models with Cross Validation
data = pd.read_csv('simple_regression.csv')  # Replace 'your_data.csv' with the actual file path

#Define the feature matrix X and target variable y
X = data[['x', 'z', 'x2']]  # Replace with the appropriate column names
y = data['y']  # Replace with the appropriate column name

#Define the models to compare and store in a tuple of name / type pairs
models = [
    ('Model 1', sm.OLS.from_formula('y ~ x + z', data)),  
    ('Model 2', sm.OLS.from_formula('y ~ x + z + x2', data))] 

#Perform cross-validation and compare models
k = 5
n = len(X)
fold_size = n // k

for model_name, model_type in models:
    rmse_scores = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size

        #Split the data into training and testing sets
        X_train = pd.concat([X[:start], X[end:]])
        y_train = pd.concat([y[:start], y[end:]])
        X_test = X[start:end]
        y_test = y[start:end]

        #Create the model
        model = model_type

        #Fit the model to the training data
        results = model.fit()

        #Predict using the testing set
        y_pred = results.predict(X_test)

        #Calculate the root mean squared error
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        rmse_scores.append(rmse)

    #Calculate the average root mean squared error across all folds for the current model
    average_rmse = np.mean(rmse_scores)

    #Print the model name and average RMSE
    print(f"{model_name}: Average RMSE = {average_rmse:.2f}")


#Stepwise Variable Selection - Forward
from itertools import combinations


#Define the feature matrix X and target variable y
X = data[['x', 'z', 'x2','y2']]  # Replace with the appropriate column names
y = data['y']  # Replace with the appropriate column name

#Perform forward stepwise variable selection
def forward_stepwise_selection(X, y, max_features=None):
    selected_features = []
    remaining_features = set(X.columns)
    
    if max_features is None:
        max_features = len(X.columns)
    
    while len(remaining_features) > 0 and len(selected_features) < max_features:
        best_pvalue = float('inf')
        best_feature = None
        
        for feature in remaining_features:
            if X[feature].dtype == 'object':
                continue
            
            model_formula = f"{feature} ~ {' + '.join(selected_features + [feature])}"
            model = sm.OLS.from_formula(model_formula, data=X.join(y)).fit()
            pvalue = model.pvalues[feature]
            
            if pvalue < best_pvalue:
                best_pvalue = pvalue
                best_feature = feature
        
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    print(selected_features)
    
forward_stepwise_selection(X, y, 2)


model = sm.OLS.from_formula('y ~ x', df)
results = model.fit()
results.summary()

#Outlier Detection
#Calculate the studentized residuals
#studentized_residuals = results.get_influence().resid_studentized_internal
#print(studentized_residuals)
#Define a threshold for outlier detection
thresholds = [1, 2, 3]  # Adjust the threshold as desired

for i in thresholds:
    studentized_residuals = results.get_influence().resid_studentized_internal
    outliers = abs(studentized_residuals) > i
    outlier_indices = data[outliers].index
    print(f"\nOutlier indices for residuals greater than {i} standard deviations:")
    print(outlier_indices)


#Outiler detection using 'Cook's distance'
sm.graphics.influence_plot(results, criterion="cooks")

#Obtain Cook's distance 
cooksd = results.get_influence().cooks_distance[0]

#Calculate critical d
critical_d = 4 / len(df)
print('Critical Cooks distance:', critical_d)

#Identify potential outliers with leverage
out_d = lm_cooksd > critical_d

#Output potential outliers with leverage
print(df.index[out_d], "\n", 
    lm_cooksd[out_d])

#Incomplete
sm.graphics.plot_partregress(
                             endog='y', # response
                             exog_i='x', # variable of interest
                             exog_others=['z','y2'], # other predictors
                             data=df,  # dataframe
                             obs_labels=True # show labels
                             );


data = pd.read_csv('mood.csv')
#Categorical variable
model = sm.OLS.from_formula('happy ~ stress + exercise', data=data).fit()

stress = data.stress
exercise = data.exercise

b0 = model.params[0]
b1 = model.params[1]
b2 = model.params[2]

happy_wo_ex = b0 + (b1 * stress) + (b2 * 0)
happy_wi_ex = b0 + (b1 * stress) + (b2 * 1) 

sns.lmplot(x='stress', y='happy', hue='exercise', fit_reg=False, data=data)
plt.plot(stress, happy_wo_ex)
plt.plot(stress, happy_wi_ex)

#Categorical interaction term
model = sm.OLS.from_formula('happy ~ stress + exercise + stress:exercise', data=data).fit()
stress = data.stress
exercise = data.exercise

b0 = model.params[0]
b1 = model.params[1]
b2 = model.params[2]
b3 = model.params[3]

happy_wo_ex = b0 + (b1 * stress) + (b2 * 0)
happy_wi_ex = b0 + (b1 * stress) + (b2 * 1) + (b3 * stress * 1)

sns.lmplot(x='stress', y='happy', hue='exercise', fit_reg=False, data=data)
plt.plot(stress, happy_wo_ex)
plt.plot(stress, happy_wi_ex)

#Continuous interaction term
model = sm.OLS.from_formula('happy ~ stress + sleep + stress:sleep', data=data).fit()
stress = data.stress
exercise = data.exercise
sleep = data.sleep

b0 = model.params[0]
b1 = model.params[1]
b2 = model.params[2]
b3 = model.params[3]

happy_sleep6 = b0 + (b1 * stress) + (b2 * 6) + (b3 * stress * 6)
happy_sleep9 = b0 + (b1 * stress) + (b2 * 9) + (b3 * stress * 9)
happy_sleep12 = b0 + (b1 * stress) + (b2 * 12) + (b3 * stress * 12)

sns.lmplot(x='stress', y='happy', fit_reg=False, data=data)
plt.plot(stress, happy_sleep6)
plt.plot(stress, happy_sleep9)
plt.plot(stress, happy_sleep12)