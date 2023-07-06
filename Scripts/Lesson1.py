#Lesson 1
#Intro to Python and Descriptive Statistics

#The pound sign is used to write comments 
#Write comments in your code so you and others can understand it

#Priting to console
print("Hi everyone!")

#Basic Arithmatic
print(123 + 456)
print(123 - 456)
print(123 * 456)
print(123 / 456)

#Basic Data Types
print(type(1))
print(type('a'))
print(type('1'))
print(type(True))
print(type('True'))

#Variables
teacher = "me"
print(teacher)

x = 5
y = 10
print(y / x)

#Lists 
example_list = [1, 2, 3, 4, 5, 6, 7]
print(example_list)

#Slicing Lists
#Lists are indexed starting at 0, meaning the first item in a list is item #0, not item #1
#The end point for a slice is not included in the returned list

print(example_list[0]) #Selects first item in list
print(example_list[1]) #Selects second number in list
print(example_list[-1]) #Selects last item in list
print(example_list[:3])  #Ends prior to 3rd place
print(example_list[2:])  #Starts at 3rd place 
print(example_list[2:4]) #Starts at 3rd place and ends before 5

#Dictionaries
example_dictionary = {'a':1, 'b':2, 'c':3}
print(example_dictionary['b'])

#Creating functions
#Declare that you're creating a function, give it a name, followed by "():"
def myfunction():
#Give your function something to do.  Also, be sure to indent.
    print("You sure put's the FUN in FUNction")
#Call your function by typing its name
myfunction()

#Creating functions with arguments / parameters
def my2ndfunction(name):
    print("Hi " + name)

my2ndfunction("Bob")

#Use "return" at the end of your function to pass data back to your script / environment
def my3rdfunction(x,y):
    return x + y

x = my3rdfunction(3, 4)
y = my3rdfunction(1, 2)
print(x)
print(y)

#"F string interpolation" 
def my4thfunction(name):
    print(f"Hi {name}!")

my4thfunction("Bob")

#For loops
list = [1, 2, 3, 4, 5, 6, 7]

for whatever in list:
    whatever = whatever ** 2
    print(whatever)


#Range
#range(start, stop, step)
#start is first number (optional, otherwise zero), stop is index point at which to stop (not including that number), step (optional) is the incrment

#Create empty lists
a = []
b = []
c = []

#Loop through the range and add each number to a list using append
for x in range(5):
    a.append(x)


for x in range(1,5):
    b.append(x)

for x in range(1,10,2):
    c.append(x)

print(a)
print(b)
print(c)

#Multiplying two lists with a loop
a = [1,2,3,4]
b = [2,3,4,5]
#Create empty list
aXb = []  

for i in range(0, len(a)):
    c = a[i] * b[i]
    aXb.append(c)    #Append adds values to a list
    
print(aXb)

#While loops
x=0

while x<20:
    print(x)
    x += 1
    

#Multiplication tables
for i in range(1, 13):
    # nested loop
    # to iterate from 1 to 10
    for j in range(1, 13):
        # print multiplication
        print(i * j, end=' ') #The end argument prevents this from printing each number to a new line


list1 = [1, 2, 3, 4, 5, 6, 7]
list2 = [2, 4, 6, 8, 10, 12, 14]

#What will this do?!?k
for x in list1:
    for y in list2:
        print(x * y, end=" ") 


#Control flow
x = 4

#If / else control flow
if x > 4:
    print("X is greater than 4")
else:
    print("X is not greater than 4")


#If / elif / else control flow
if x > 4:
    print("X is greater than 4")
elif x < 4:
    print("X is less than 4")
else: 
    print("x must be four if it isn't greater or less")

#Length with len()
v = [4, 2, 6, 11, 9, 3, 8]
n = len(v)

#The sorted() function returns a list sorted from lowest to highest value
s = sorted(v)
print(s)

#Function to calculate mean
def calculate_mean(values):
    return sum(values) / len(values) #the sum function takes the sum...

#The modulus, %, returns the remainder after division  
print(26 % 4) # 4 goes into 26 6 whole times with a remainder of two (6x4=24+2=26)
print(12 % 4) # 4 goes into 12 3 whole times, with 0 remainder 

#If the series is even numbered, lenght % 2 = 0, if odd numbered length % 2 = 1
n = 4
print(n % 2)
print((n + 1) % 2)

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


#Import Panadas so we can import and manipulate data
import pandas as pd
#Load data into a dataframe, then view the first 10 rows
df = pd.read_csv("monthly_sales.csv")
df.head(10)

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

#Even easier
df.describe()

#Import matplotlib for graphics
import matplotlib.pyplot as plt

#Loop through each column and create a histogram
for column in columns:
    plt.hist(df[column], bins=10, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

#Generating Random Data with Numpy
#Generate 10 values between zero and 100) from a uniform distribution
x = np.random.randint(100, size=(10))
print(x)
print(type(x)) #Output is in the form of Numpy arrays, not lists

#Generate 10 valeus between 50 and 100
y = np.random.randint(50, 100, size=(10))
print(y)

#Generate 2 sets of 10 values between 1 and 100
x = np.random.randint(100, size=(10,2))
print(x)

#Convert the Numpy array into a Pandas dataframe
df = pd.DataFrame(x, columns = ['A','B'])
df

#Generating Normally Distributed Random Data
# loc = mean, scale = standard deviation, size = sample size (n)
n = np.random.normal(loc=0, scale=1, size=(10))
print(n)

#For uniform, low = lowest number, high = highest number, size = sample size
u = np.random.uniform(low=1, high=100, size=10)
print(u)

#Generating Binomial and Poisson Distributed Data

#For binomial,n = number of trials, p = probability of success for a given trial, size = sample size
b = np.random.binomial(n=50, p=0.25, size=10)
print(b)

#For poisson, lam = rate of occurrance, size = sample size
p = np.random.poisson(lam=5, size=10)
print(p)

import seaborn as sns
sns.displot(p)
sns.displot(b)
sns.displot(n)
sns.displot(u)