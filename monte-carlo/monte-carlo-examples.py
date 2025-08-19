#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import logging
logger = logging.getLogger(__name__)



def monte_carlo_pi(n_samples=1_000):
    # Generate random (x, y) points in [0, 1] × [0, 1]
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    
    # Count how many points fall inside the quarter circle (x² + y² ≤ 1)
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.mean(inside_circle)
    
    return pi_estimate

def monte_carlo_mean_median(n_samples=100, n_trials=10):
    diffs = []
    for _ in range(n_trials):
        data = np.random.chisquare(df=1.0, size=n_samples)
        logger.debug(data)

        mu = np.mean(data)
        m = np.median(data)
        epsilon = abs(mu - m)
        diffs.append(epsilon)
    return np.mean(diffs), np.std(diffs)

def monte_carlo_dice(n_dice=2, n_sides=6, n_trials=100_000):
    throws = np.random.randint(1, high=n_sides+1, size=(n_trials, n_dice))
    logger.debug(throws)
    return np.mean(throws), np.std(throws), throws


def initiate_logging():
    logging.basicConfig(filename='monte-carlo-examples.log', level=logging.INFO)

if __name__ == '__main__':
    initiate_logging()

    # print("Monte Carlo estimate of π:", monte_carlo_pi())
    # 
    # avg_diff, std_diff = monte_carlo_mean_median()
    # print(f"Monte Carlo Average |µ - m| = {avg_diff:.4f} ± {std_diff:.4f}")
    
    avg_throw, std_throw, throws = monte_carlo_dice()
    
    print(f"Monte Carlo dice |µ - m| = {avg_throw:.4f} ± {std_throw:.4f}")
