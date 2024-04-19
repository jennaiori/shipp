'''Module containing unit tests for the package'''

from sizing_opt_hpp.components import Storage, Production, OpSchedule
from sizing_opt_hpp.timeseries import TimeSeries


def test_simple():
    '''
        Dummy test to set up continuous integration pipeline
    '''
    data = [1, 1, 1, 1]
    dt = 1
    ts = TimeSeries(data, dt)

    assert ts.std() == 0
    assert ts.mean() == 1

    ts_empty =TimeSeries()

    print('empty',ts_empty.std())
    print('empty',ts_empty.mean())

    prod_unit = Production(ts, 0)
    cost_prod = prod_unit.get_tot_costs()

    # assert ts.dt == prod_unit.power.dt
    stor_unit = Storage()
    stor_p = TimeSeries([1,1,1,1], dt)
    stor_e = TimeSeries([0, 1, 2, 3], dt)

    os = OpSchedule( [prod_unit], [stor_unit], [prod_unit.power], [stor_p], [stor_e])
    print(os)

    # os.plot_powerflow()