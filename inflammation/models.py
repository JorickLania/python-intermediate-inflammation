"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2 dimensional array) where each row contains
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):  
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array for each day.

    :param data: 2D array with inflammation data, rows containing single patient
                 measurements for all days.
    :returns:    Array of mean values of measurements for each day."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily maximum of a 2D inflammation data array for each day.

    :param data: 2D array with inflammation data, rows containing single patient
                 measurements for all days.
    :returns:    Array of max values of measurements for each day."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily minimum of a 2D inflammation data array for each day.

    :param data: 2D array with inflammation data, rows containing single patient
                 measurements for all days.
    :returns:    Array of min values of measurements for each day."""
    return np.min(data, axis=0)


def daily_standard_deviation(data):
    """Computes and returns standard deviation for data."""
    mean = np.mean(data, axis=0)
    devs = []
    for entry in data:
        devs.append((entry - mean) * (entry - mean))

    return np.std(data, axis=0)
