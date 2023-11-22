'''
    Module used to define "kernel" functions, i.e. functions that are 
    used throughout the code for various calculations.

'''

import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

def power_calc(wind, radius, cp, v_in, v_r, v_out):
    '''
        Function to calculate the power output of wind based on the wind
        data (wind), the rotor radius (radius), the power coefficient (cp),
        the cut-in, rated and cut-out wind speed (v_in, v_r, v_out)
        The function output the power time series (power) in MW
    '''
    rho = 1.225
    pi = 3.141591
    power = wind**3 * 0.5 * rho * pi * radius **2 * cp * 1e-6
    power[np.logical_or(wind<v_in,wind>v_out)] = 0
    power[wind>= v_r] = v_r**3 * 0.5 * rho * pi * radius **2 * cp * 1e-6
    return power

def plot_xcorr(x, y, delta_t = 1/24):
    '''
        Plot cross-correlation (full) between two signals.
    '''
    n_max = max(len(x), len(y))
    n_min = min(len(x), len(y))

    if n_max == len(y):
        lags = np.arange(-n_max + 1, n_min)
    else:
        lags = np.arange(-n_min + 1, n_max)
    c = correlate((x - np.mean(x)) / np.std(x), (y - np.mean(y))  / np.std(y), 'full')

    plt.plot(lags*delta_t, c / n_min)
    plt.xlim([delta_t, max(lags*delta_t)])
    plt.xscale('log')
    plt.xlabel('Time lag [days]')
    plt.ylabel('Cross-correlation coefficient [-]')
