#The "Law of Large Numbers" shows us that the larger our sample is, relative to the population, the closer our sample mean will be to the population mean / expected value

import numpy as np
import matplotlib.pyplot as plt

#Generate normally distributed data
mean = 100
std_dev = 25
size = 10000
data = np.random.normal(mean, std_dev, size)
population_mean = np.mean(data)
print(population_mean)

#Sample the data
sample_sizes = [10, 50, 100, 250, 500, 750, 999, 2500 ,5000, 7500, 9000, 9999]
#Create empty lists to store calculations
means = []
differences = []

#Loop through the sample_sizes list and take a sample of each size
#Calculate the mean of each sample, then append it to the means list
#Calculate how far the each sample mean is from the population mean, then append it to the differences list
for i in sample_sizes:
    sample = np.random.choice(data, size=i)
    mean = np.mean(sample)
    print(f"The sample mean is {mean}\n")
    means.append(mean)
    dif = abs(population_mean - mean)
    print(f"The difference between the population and sample mean is {dif} when the sample size is {i} \n")
    differences.append(dif)

#Zip the two lists into a "Zip object"
sizes_and_diffs_zip = zip(sample_sizes, differences) 
#Convert the "Zip object" into a list of tuples, pairs of size and corresponding differences
sizes_and_diffs_list = list(sizes_and_diffs_zip)
print(sizes_and_diffs_list)

#Plot the sample size vs the error (estimated mean vs actual)
plt.scatter(sample_sizes, differences)
plt.plot(sample_sizes, differences, '-', color='red', label='Line')
plt.title('Sample Size vs Error')
plt.show()