'''
    Module containing unit tests for the package
'''

from ..components import Storage, Production, OpSchedule
from ..timeseries import TimeSeries


def test_simple():
    '''
        Dummy test to set up continuous integration pipeline
    '''
    data = [1, 1, 1, 1]
    dt = 1
    ts = TimeSeries(data)

    assert ts.std() == 0
    assert ts.mean() == 1

    ts_empty =TimeSeries()

    print('empty',ts_empty.std())
    print('empty',ts_empty.mean())

    prod_unit = Production(0.5, 2, 0)
    prod_p = prod_unit.get_production(ts)

    stor_unit = Storage()
    stor_p = TimeSeries([1,1,1,1], dt)
    stor_e = TimeSeries([0, 1, 2, 3], dt)

    os = OpSchedule( [prod_unit], [stor_unit], [prod_p], [stor_p], [stor_e])
    print(os)

    # os.plot_powerflow()
