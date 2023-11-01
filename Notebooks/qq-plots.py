import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(7134)
sample_size = 1000

uniform_data = np.random.uniform(0, 1, sample_size)
normal_data = np.random.normal(0, 1, sample_size)
exponential_data = np.random.exponential(1, sample_size)
gamma_data = np.random.gamma(2, 1, sample_size)

plt.figure(figsize=(16, 6))

#Uniform QQ plot
plt.subplot(241)
stats.probplot(uniform_data, dist="norm", plot=plt)
plt.title("Uniform QQ Plot")

#Uniform Histogram
plt.subplot(245)
plt.hist(uniform_data, bins=20, density=True, alpha=0.7)
plt.title("Uniform Histogram")

#Normal QQ plot
plt.subplot(242)
stats.probplot(normal_data, dist="norm", plot=plt)
plt.title("Normal QQ Plot")

#Normal Histogram
plt.subplot(246)
plt.hist(normal_data, bins=20, density=True, alpha=0.7)
plt.title("Normal Histogram")

#Exponential QQ plot
plt.subplot(243)
stats.probplot(exponential_data, dist="norm", plot=plt)
plt.title("Exponential QQ Plot")

#Exponential Histogram
plt.subplot(247)
plt.hist(exponential_data, bins=20, density=True, alpha=0.7)
plt.title("Exponential Histogram")

#Gamma QQ plot
plt.subplot(244)
stats.probplot(gamma_data, dist="norm", plot=plt)
plt.title("Gamma QQ Plot")

#Gamma Histogram
plt.subplot(248)
plt.hist(gamma_data, bins=20, density=True, alpha=0.7)
plt.title("Gamma Histogram")

plt.tight_layout()
plt.show()
