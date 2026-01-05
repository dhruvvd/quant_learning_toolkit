import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.stats import skew, kurtosis

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

def sample_exponential(rate: float, n: int = 10000) -> np.ndarray:
    x = np.random.uniform(0.0, 1.0, size=n)
    y_sampled = (-1 / rate) * np.log(1 - x)
    return y_sampled

def sample_lognormal(MEAN: float, STD: float, n: int = 10000):
    x = np.random.uniform(0.0, 1.0, size=n)
    y_sampled = np.exp(MEAN + (np.sqrt(2 * STD**2) * erfinv(2*x - 1)))
    return y_sampled


# ------------ STATISTICAL ANALYSIS FUNCTIONS ------------

def calculate_moments(x: np.ndarray) -> dict:
    mean = x.mean()
    var = (np.std(a=x))**2
    skewness = skew(x)
    kurt = kurtosis(x)

    moments = {
        "Mean": mean,
        "Variance": var,
        "Skewness": skewness,
        "Kurtosis": kurt
    }

    return moments
        
def empirical_cdf(x: np.ndarray, t: float):
    return np.count_nonzero(x <= t) / len(x)

def pdf_estimate(x: np.ndarray, type: str, info: dict) -> np.ndarray:
    pdf = []

    match type.lower():
        case "uniform":
            for samp in x:
                if info['min'] < samp and info['max'] > samp:
                    pdf.append(1 / info['max'] - info['min'])
            return pdf
        case "normal":
            pdf = (1 / info['std'] * (np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - info['mean']) / info['std'])**2)
            return pdf
        case "exponential":
            pdf = info['rate'] * np.exp(-1 * info['rate'] * x)
            return pdf
        case "lognormal":
            pdf = (1 / (x * info['std']) * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (np.log(x) - info['mean'] / info['std'])**2)
            return pdf


# ------------ VISUALIZATION FUNCTIONS ------------

def plot_histogram():

def plot_qq():

def compare_distributions():