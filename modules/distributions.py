import numpy as np
import matplotlib.pyplot as plt

# ------------ SAMPLING FUNCTIONS ------------

def sample_uniform(MIN: float, MAX: float, n: int = 10000) -> np.ndarray:
    x = np.random.uniform(0.0, 1.0, size=n)
    y_sampled = MIN + (x * (MAX - MIN))
    return y_sampled


def sample_normal(MEAN: f):

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