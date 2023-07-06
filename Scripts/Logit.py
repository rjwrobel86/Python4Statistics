import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('logit_data.csv')
data.columns
#Uncomment lines below to create random data for testing purposes 
#n = 1000
#data['gender'] = np.random.choice(['Male', 'Female'], size=n)
#data['age'] = np.random.randint(18, 65, size=n)
#data['visits'] = np.random.randint(0, 10, size=n)
#data['past_purchases'] = np.random.randint(0, 10, size=n)
#data['sold'] = np.random.randint(0, 1, size=n)

X = data[['gender', 'age', 'visits', 'past_purchases']] #Create dataframe for indpenedent variables
X = pd.get_dummies(X, drop_first=True)  #Convert gender variable to dummy variables
X = sm.add_constant(X)  #Add a constant term to the independent variables
y = data['purchased'] #Create dataframe for dependent variable

#Fit the model
model = sm.Logit(y, X)
result = model.fit()

#Model results
print(result.summary())

yhat_probabilities = result.predict(X) #Get probabilities for each observation
yhat = np.round(yhat_probabilities) #Round probabilities to 0 (no) or 1 (yes)
yhat = yhat.astype(int) #Float to Int 
yhat.head()

#Calculate marginal effects
marginal_effects = result.get_margeff(at='mean')
print("Marginal Effects:")
print(marginal_effects.summary())
#Marginal Effect: To obtain the marginal effect of a predictor variable, you can calculate the derivative of the predicted probabilities with respect to that variable. 
#The marginal effect represents the change in the predicted probability of the positive outcome due to a one-unit change in the predictor variable.

elasticities = result.get_margeff(at='mean',method='eyex')
print(elasticities.summary())
#Interpret as % change in Y from a 1% increase in X

coefficients = result.params
odds_ratios = np.exp(coefficients)

print("Odds Ratios:")
print(odds_ratios)
#Exponentiating a logit coefficient provides the odds ratio associated with a one-unit increase in the corresponding predictor variable. 
#For example, if the exponentiated coefficient is 3, it indicates that a one-unit increase in the predictor variable is associated with a trippling of the odds of the positive outcome.

accuracy = accuracy_score(y, yhat)
print("Accuracy:", accuracy)

report = classification_report(y, yhat)
print("Classification Report:")
print(report)

#Precision is the ratio of true positive predictions to the total number of positive predictions made by the model.
#Precision is concerned with the correctness of positive predictions 

#Recall is the ratio of true positive predictions to the total number of actual positive instances in the dataset.
#Recall is concerned with avoiding false negatives.

#F1-Score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. 
#The F1 score considers both false positives and false negatives and is useful when you want to find a balance between precision and recall.

#Support is the number of actual instances of each outcome in the dataset

#Prediction on new data
new_data = pd.DataFrame({'gender_M': ['Male', 'Female','Male'],
                         'age': [30, 40, 50],
                         'visits': [5, 8, 2],
                         'past_purchases': [2, 6, 0]})

new_data = pd.get_dummies(new_data, drop_first=True)  
new_data = sm.add_constant(new_data)  
yhat_probabilities = result.predict(new_data)
yhat = np.round(yhat_probabilities)  
yhat = yhat.astype(int)

print("Predictions:", yhat)
print("Probabilities:", yhat_probabilities)

#Coefficient Plot
coefficients = result.params
conf_int = result.conf_int()

plt.errorbar(coefficients.index, coefficients, yerr=(coefficients - conf_int[0], conf_int[1] - coefficients), fmt='o', capsize=5)
plt.axhline(0, color='gray', linestyle='--')  # Add a reference line at 0
plt.xlabel('Independent Variables')
plt.ylabel('Coefficients')
plt.title('Coefficient Plot')
plt.xticks(rotation=45)
plt.show() 

#Calculate column means
mean_age = X['age'].mean()
mean_visits = X['visits'].mean()
mean_past_purchases = X['past_purchases'].mean()
mean_gender = X['gender_M'].mean()
n = len(X)

#Generate span variables
age_span = np.linspace(X['age'].min(), X['age'].max(), len(X))
visits_span = np.linspace(X['visits'].min(), X['visits'].max(), len(X))
past_purchases_span = np.linspace(X['past_purchases'].min(), X['past_purchases'].max(), len(X))
gender_M_span = np.linspace(X['gender_M'].min(), X['gender_M'].max(), len(X))

#Subsitute means for spans
#Here age is used as an example
df2 = pd.DataFrame({
    'const': np.full(n, 1),
    'age': np.full(n, mean_age),
    'visits': np.full(n, mean_visits),
    'past_purchases': np.full(n, mean_past_purchases),
    'gender': np.full(n, mean_gender)
})

yhat_probabilities = result.predict(df2)
yhat = np.round(yhat_probabilities)  
yhat = yhat.astype(int)

plt.plot(yhat_probabilities)

#Bulk probability plots

column_dict = {'age':age_span,'visits':visits_span,'past_purchases':past_purchases_span,'gender':gender_span}

num_rows = (len(column_dict) + 1) // 2

fig, axs = plt.subplots(num_rows, 2, figsize=(10, 8))

#Flatten the axs array if it has multiple dimensions
if num_rows > 1:
    axs = axs.flatten()
    
for i, column in enumerate(column_dict):
    original_column = df2[column].copy()
    df2[column] = column_dict[column]
    yhat_probabilities = result.predict(df2)

    # Plot the subplot
    axs[i].plot(df2[column], yhat_probabilities)
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Probability of Purchase')
    axs[i].set_title(f'Plot for {column}')

    df2[column] = original_column

plt.tight_layout() #Better spacing
plt.show()

params = result.params
intercept = params[0]
age_coef = params[1]
visits_coef = params[2]
past_purchases_coef = params[3]
gender_coef = params[4]

def predict_probability(age, visits, past_purchases, gender):
    if gender == "m":
        gender = 1
    elif gender == "f":
        gender = 0
    else:
        gender = gender
        
    
    # Step 1: Compute logit score
    logit_score = intercept + age_coef*age + visits_coef*visits + past_purchases_coef*past_purchases + gender_coef*gender
    
    # Step 2: Convert logit score to a probability
    probability = 1 / (1 + np.exp(-logit_score))
    
    if gender == 1:
        gender_var = "male"
    else:
        gender_var = "female"
    
    print(f'The probability that someone age {age}, who has vistied {visits} times, purchased {past_purchases} time(s) in the past, and is {gender_var}, will purchase is  {probability}')
    
    return probability

predict_probability(25,1,1,"f")
predict_probability(25,1,1,"m")