'''
    module timeseries

    contains the class TimeSeries
'''

import numpy as np

class TimeSeries:
    '''
        class TimeSeries
        Used to encapsulate data in the time domain associated to the 
        time step. The class provides functions for the statistical 
        analysis of time series.
    '''

    def __init__(self, data=None, dt=0):
        if data is None:
            self.data = None
        else:
            self.data = np.array(data)

        self.dt = dt

    def __str__(self):
        return f"TimeSeries instance: data is {self.data}, dt is {self.dt}"

    def std(self):
        '''Compute the standard deviation of the data'''
        if self.data is None:
            return None

        return np.std(self.data)

    def mean(self):
        '''Computes the mean value of the data'''
        if self.data is None:
            return None

        return np.mean(self.data)

    def time(self):
        '''Computes the time vector with increment dt'''
        return np.arange(0, self.dt*len(self.data), self.dt)
