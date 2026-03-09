import numpy as np
from scipy.stats import poisson, binom
from scipy.special import factorial, jve, i0
from math import sqrt, exp

# --------------------------
# Exact Bessel probability
# --------------------------
def exact_bessel_dist_prob(i, n):
    arg = 2 * sqrt((n - i) * (i - 1))
    total = 0
    if arg == 0:
        # Edge case, i=1 or i=n
        poisson_dist = poisson(n - 1)
        for k in range(1, 21):
            delta = (0.5 * poisson_dist.pmf(k) + poisson_dist.cdf(k - 1)) / factorial(k)
            total += delta
        return exp(-1) * total
    term = 1  # 1/0!
    for eta in range(1, 21):
        prefix = ((n - i) / (i - 1)) ** (eta / 2) + ((i - 1) / (n - i)) ** (eta / 2)
        term += 1 / factorial(eta)
        delta = (0.5 / factorial(eta) + exp(1) - term) * prefix * jve(eta, arg)
        total += delta
    return exp(arg - n) * ((exp(1) - 1) * i0(arg) + total)

# --------------------------
# Approximate Bessel probability
# --------------------------
def approx_bessel_dist_prob(i, n):
    arg = 2 * sqrt((n + 1 - i) * i)
    return 2 * exp(arg - n - 1) * i0(arg)

# --------------------------
# Distribution classes
# --------------------------
class PoissonBootstrapExactSampleMedianDistribution:
    def __init__(self, n):
        self.n = n
        self.cdf = np.zeros(n)
        total = 0
        for i in range(1, n + 1):
            total += exact_bessel_dist_prob(i, n) / (1 - exp(-n))
            self.cdf[i - 1] = min(1, total)

    def pdf(self, i):
        return exact_bessel_dist_prob(i, self.n)

    def cdf_val(self, i):
        return self.cdf[i - 1]

    def quantile(self, q):
        if q > self.cdf[-1]:
            return self.n
        return np.searchsorted(self.cdf, q) + 1

    def rvs(self, size=1):
        """Генерация случайных индексов с вероятностью, заданной CDF"""
        u = np.random.uniform(0, 1, size)
        return np.searchsorted(self.cdf, u) + 1


class PoissonBootstrapApproximateSampleMedianDistribution:
    def __init__(self, n):
        self.n = n
        self.cdf = np.zeros(n)
        total = 0
        for i in range(1, n + 1):
            total += approx_bessel_dist_prob(i, n) / (1 - exp(-n))
            self.cdf[i - 1] = min(1, total)

    def pdf(self, i):
        return approx_bessel_dist_prob(i, self.n)

    def cdf_val(self, i):
        return self.cdf[i - 1]

    def quantile(self, q):
        if q > self.cdf[-1]:
            return self.n
        return np.searchsorted(self.cdf, q) + 1

    def rvs(self, size=1):
        """Генерация случайных индексов с вероятностью, заданной CDF"""
        u = np.random.uniform(0, 1, size)
        return np.searchsorted(self.cdf, u) + 1

class PoissonBootstrapApproximateSampleMedianDistribution:
    def __init__(self, n):
        self.n = n
        self.cdf = np.zeros(n)
        total = 0
        for i in range(1, n + 1):
            total += approx_bessel_dist_prob(i, n) / (1 - exp(-n))
            self.cdf[i - 1] = min(1, total)

    def pdf(self, i):
        return approx_bessel_dist_prob(i, self.n)

    def cdf_val(self, i):
        return self.cdf[i - 1]

    def quantile(self, q):
        if q > self.cdf[-1]:
            return self.n
        return np.searchsorted(self.cdf, q) + 1

    def rvs(self, size=1):
        """Генерация случайных индексов с вероятностью, заданной CDF"""
        u = np.random.uniform(0, 1, size)
        return np.searchsorted(self.cdf, u) + 1

# --------------------------
# Confidence interval function
# --------------------------
def confidence_interval(x, x_quantile_dist, y, y_quantile_dist, alpha, B):
    sample1_quantile_indexes = x_quantile_dist.rvs(B)
    sample2_quantile_indexes = y_quantile_dist.rvs(B)

    min_x, max_x = sample1_quantile_indexes.min(), sample1_quantile_indexes.max()
    min_y, max_y = sample2_quantile_indexes.min(), sample2_quantile_indexes.max()

    ordered_x = np.partition(x, np.arange(min_x-1, max_x))
    ordered_y = np.partition(y, np.arange(min_y-1, max_y))

    diff_in_quantile = ordered_y[sample2_quantile_indexes - (min_y - 1)] - \
                       ordered_x[sample1_quantile_indexes - (min_x - 1)]

    return np.quantile(diff_in_quantile, [alpha/2, 1-alpha/2])

# --------------------------
# Two-sample bootstrap wrappers
# --------------------------
def two_sample_poisson_bootstrap_binomial_quantile_confidence_interval(x, y, alpha, q, B):
    return confidence_interval(x, binom(len(x)+1, q), y, binom(len(y)+1, q), alpha, B)

def two_sample_poisson_bootstrap_approximate_median_confidence_interval(x, y, alpha, B):
    return confidence_interval(x,
                               PoissonBootstrapApproximateSampleMedianDistribution(len(x)),
                               y,
                               PoissonBootstrapApproximateSampleMedianDistribution(len(y)),
                               alpha, B)

def two_sample_poisson_bootstrap_exact_median_confidence_interval(x, y, alpha, B):
    return confidence_interval(x,
                               PoissonBootstrapExactSampleMedianDistribution(len(x)),
                               y,
                               PoissonBootstrapExactSampleMedianDistribution(len(y)),
                               alpha, B)

# --------------------------
# Example usage
# --------------------------
from scipy.stats import norm

alpha = 0.05
N1 = 10000
N2 = 10000
B = 100

x = norm(10, 2).rvs(N1)
y = norm(5, 8).rvs(N2)

mu1, sigma1 = 10, 2
sample_normal1 = np.random.normal(mu1, sigma1, N1)

# Теоретическая медиана для нормального распределения = μ
theoretical_median_normal1 = mu1

mu2, sigma2 = 5, 8
sample_normal2 = np.random.normal(mu2, sigma2, N1)

sample_normal1_sorted = np.sort(sample_normal1)
sample_normal2_sorted = np.sort(sample_normal2)

print(two_sample_poisson_bootstrap_binomial_quantile_confidence_interval(sample_normal1_sorted, sample_normal2_sorted, alpha, 0.5, B))
# print(two_sample_poisson_bootstrap_approximate_median_confidence_interval(sample_normal1, sample_normal2, alpha, B))
# print(two_sample_poisson_bootstrap_exact_median_confidence_interval(x, y, alpha, B))
