import numpy as np
import matplotlib.pyplot as plt

#Generate normally distributed data
mean = 100
std_dev = 25
size = 10000
data = np.random.normal(mean, std_dev, size)

#Create an empty list to store sample means
means = []

#Take 100 samples of 20 observations from the population data
for i in range(100):
    sample = np.random.choice(data, size=20)
    mean = np.mean(sample)
    means.append(mean)
    
#Plot the original distribution
plt.hist(data)
plt.title('Normally Distributed Data, n = 10000')
plt.show()
    
#Plot the distribution of sample means 
plt.hist(means)
plt.title("100 Sample Means from some Normally Distributed Data")
plt.show()

#Do the same with data from a uniform distribution (all numbers equally likely)
uniform_data = np.random.uniform(low=0, high=10, size=10000)
u_means = []

for i in range(1000):
    sample = np.random.choice(uniform_data, size=20)
    mean = np.mean(sample)
    u_means.append(mean)
    

plt.hist(uniform_data, bins=50)
plt.title('Uniformely Distributed Data, n = 10000')
plt.show()

plt.hist(u_means, bins=50)
plt.title("100 Sample Means from some Uniform Distributed Data")
plt.show()

#Do the same with data from a poisson distribution
poisson_data = np.random.poisson(lam=3, size=10000)
p_means = []

for i in range(100):
    sample = np.random.choice(poisson_data, size=20)
    mean = np.mean(sample)
    p_means.append(mean)
    

plt.hist(poisson_data, bins=20)
plt.title('Pooisson Distributed Data, n = 10000')
plt.show()

plt.hist(p_means, bins=20)
plt.title("100 Sample Means from some Poisson Distributed Data")
plt.show()

#Do the same with data from a geometric distribution 
geometric_data = np.random.geometric(p=0.35, size=10000)
g_means = []

for i in range(100):
    sample = np.random.choice(geometric_data, size=i)
    mean = np.mean(sample)
    g_means.append(mean)
    
plt.hist(geometric_data, bins=20)
plt.title('Geometrically Distributed Data, n = 10000')
plt.show()

plt.hist(g_means, bins=20)
plt.title("100 Sample Means from some Geometric Distributed Data")
plt.show()