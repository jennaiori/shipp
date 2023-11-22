'''
    Module containing unit tests for the package
'''

from ..timeseries import TimeSeries

def test_simple():
    '''
        Dummy test to set up continuous integration pipeline
    '''
    data = [1, 1, 1, 1]
    ts = TimeSeries(data)

    assert ts.std() == 0
    assert ts.mean() == 1

    ts_empty =TimeSeries()

    print('empty',ts_empty.std())
    print('empty',ts_empty.mean())
