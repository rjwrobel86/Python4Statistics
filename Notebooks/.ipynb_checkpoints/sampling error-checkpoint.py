import numpy as np
import matplotlib.pyplot as plt

mean = 100
std_dev = 20
data = np.random.normal(mean, std_dev, 10000)

sample_size = 25
number_of_samples = 100
sample_means = [np.mean(np.random.choice(data, sample_size, replace=False)) for _ in range(number_of_samples)]

print(np.mean(data))
print(np.mean(sample_means))
print(max(sample_means))
print(min(sample_means))


plt.hist(sample_means, bins=10, edgecolor='black')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Means (n=25)')
plt.show()
