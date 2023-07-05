#Import libraries
import numpy as np
import matplotlib.pyplot as plt

#Generate fake data
#Set a random seed for reproducibility
np.random.seed(7134)
x = np.linspace(0, 10, 100) #Using linspace gives even distanced values
y = np.sin(x) + np.random.normal(0, 0.2, 100) #Using normal gives normally distributed values
z = [np.random.normal(0, std, 100) for std in range(1, 5)] #Using normal again, but with a list comprehension
categories = ['A', 'B', 'C', 'D', 'E'] #Categorical variables
values = np.random.randint(1, 10, len(categories)) #Using randint gives random integer values

#Line plot
plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title('Line Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Bar plot
plt.figure(figsize=(8, 4))
plt.bar(categories, values)
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()

#Histogram
plt.figure(figsize=(8, 4))
plt.hist(y, bins=20)
plt.title('Histogram')
plt.xlabel('y')
plt.ylabel('Frequency')
plt.show()

#Box plot
plt.figure(figsize=(8, 4))
plt.boxplot(z)
plt.title('Box Plot')
plt.xlabel('Data')
plt.ylabel('Value')
plt.show()

#Pie chart
plt.figure(figsize=(8, 4))
plt.pie(values, labels=categories, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()