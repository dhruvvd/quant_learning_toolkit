import numpy as np
import matplotlib.pyplot as plt

# ------------ SAMPLING FUNCTIONS ------------

def sample_uniform(MIN: float, MAX: float, n: int = 10000) -> np.ndarray:
    x = np.random.uniform(0.0, 1.0, size=n)
    y_sampled = MIN + (x * (MAX - MIN))
    return y_sampled


def sample_normal(MEAN: float, STD: float, n: int = 10000) -> np.ndarray:
    def normal_dist(x: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (
            1
            / (std * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - mean) / std) ** 2)
        )
    
    def laplace_dist(x: np.ndarray, mean: float, scale: float) -> np.ndarray:
        return (1 / 2 * scale) * np.exp(-1 * abs(x - mean) / scale)


    x = np.random.laplace(loc=MEAN, size=n)
    k = 1.5
    u = np.random.uniform(0, laplace_dist(x, MEAN, STD) * k)
    (idx,) = np.where(u < normal_dist(x, mean=MEAN, std=STD))
    return x[idx], len(idx) / n

def sample_exponential():

def sample_lognormal():


# ------------ STATISTICAL ANALYSIS FUNCTIONS ------------

def calculate_moments():

def empirical_cdf():

def pdf_estimate():


# ------------ VISUALIZATION FUNCTIONS ------------

def plot_histogram():

def plot_qq():

def compare_distributions():