'''Module containing unit tests for the module kernel.py'''

import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo

def test_solve_lp_pyomo():
    '''Test of function solve_lp_sparse'''
    power = np.array([2,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_min = 0.5
    p_min_vec = np.array([0.5,0.5,0.5,0.5,0.5])
    p_max = 4.0

    power_ts = TimeSeries(0.5*power, dt)
    price_ts = TimeSeries(price, dt)
    stor_batt = Storage(1,1,1,1,1,1)
    stor_h2 = Storage(1,1,1,1,1,1)
    discount_rate = 0.03
    n_year = 20

    prod_wind = Production(power_ts, p_cost = 1)
    prod_pv = Production(power_ts, p_cost = 1)

    _ = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min, p_max, n)

    _ = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min_vec, p_max, n)

    try:
        _ = solve_lp_pyomo(TimeSeries(price, 2*dt), prod_wind, prod_pv,
                            stor_batt, stor_h2, discount_rate, n_year,
                            p_min_vec, p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    # Raise an error if the optimization does not converge
    try:
        _ = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                           discount_rate, n_year, 2.0, 0.0, n)
    except RuntimeError:
        assert True
    else:
        assert False
