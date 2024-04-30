'''Module containing unit tests for the module kernel.py'''

import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel import build_lp_obj_npv_sf, build_lp_cst_sparse_sf, \
                            solve_lp_sparse_sf


def test_build_lp_obj_npv():
    ''' Test of function build_lp_obj_npv_sf'''

    power = np.array([0,1,2,3])
    n = len(power)

    price = np.array([1, 1, 1, 1])
    discount_rate = 0.3
    n_year = 20

    vec_obj = build_lp_obj_npv_sf(price, n, 1,1,1,1, discount_rate, n_year)

    n_x = 4*n+6

    assert vec_obj.ndim == 1
    assert vec_obj.shape[0] == n_x

    try:
        vec_obj = build_lp_obj_npv_sf(price,  n, 1,1,1,1, 1.2, n_year)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_lp_obj_npv_sf(price,  n, 1,1,1,1, discount_rate, -1)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_lp_obj_npv_sf(price[:n-1], n,  1,1,1,1, discount_rate,
                                   n_year)
    except AssertionError:
        assert True
    else:
        assert False


def test_build_lp_cst_sparse():
    ''' Test of function build_lp_cst_sparse_sf'''

    power = np.array([0,1,2, 0, 3])
    n = len(power)
    dt = 1.0
    p_min = 1.0
    p_min_vec = np.array([0,0,0,1,0])
    p_max = 4.0
    losses_batt = 0.0
    losses_h2 = 0.0
    rate_batt = 1.0
    rate_h2 = 1.0
    max_soc = 1.0
    max_h2 = 1.0

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse_sf(power, dt, p_min, p_max, n, losses_batt, losses_h2)
    n_x = 4*n+6
    n_eq = 2
    n_ineq = 12*n+2

    assert mat_eq.shape[0] == n_eq
    assert mat_eq.shape[1] == n_x
    assert vec_eq.shape[0] == n_eq
    assert mat_ineq.shape[0] == n_ineq
    assert mat_ineq.shape[1] == n_x
    assert vec_ineq.shape[0] == n_ineq
    assert lb.shape[0] == n_x
    assert ub.shape[0] == n_x
    assert vec_eq.ndim == 1
    assert vec_ineq.ndim == 1
    assert lb.ndim == 1
    assert ub.ndim == 1

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse_sf(power, dt, p_min_vec, p_max, n, losses_batt,
                            losses_h2)

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse_sf(power, dt, p_min_vec, p_max, n, losses_batt,
                            losses_h2, rate_batt, rate_h2, max_soc, max_h2)

    try:
        mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
            build_lp_cst_sparse_sf(power[:n-1], dt, p_min_vec, p_max, n,
                                losses_batt, losses_h2)
    except AssertionError:
        assert True
    else:
        assert False




def test_solve_lp_sparse():
    '''Test of function solve_lp_sparse_sf'''
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

    _ = solve_lp_sparse_sf(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min, p_max, n)

    _ = solve_lp_sparse_sf(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min_vec, p_max, n)

    try:
        _ = solve_lp_sparse_sf(TimeSeries(price, 2*dt), prod_wind, prod_pv,
                            stor_batt, stor_h2, discount_rate, n_year,
                            p_min_vec, p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        _ = solve_lp_sparse_sf(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                                    discount_rate, n_year, 4.0, 0.0, n)
    except RuntimeError:
        assert True
    else:
        assert False



