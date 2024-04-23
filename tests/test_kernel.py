'''Module containing unit tests for the module kernel.py'''

import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel import power_calc, build_lp_cst, build_lp_obj_pareto, \
                        build_lp_obj_npv, build_milp_obj, build_lp_cst_sparse,\
                        build_milp_cst_sparse, solve_lp, linprog_mosek, \
                        milp_mosek, solve_lp_sparse_pareto, solve_lp_sparse,\
                        solve_lp_sparse_old, solve_milp_sparse



def test_power_calc():
    ''' Test of function power_calc'''

    wind = np.array([1.0, 2.0, 2.0, 3.0])
    radius = 1.0
    cp = 0.4
    v_in = 1.0
    v_out = 2.5
    v_r = 2.0
    p_max = 1.0
    power = power_calc(wind, radius, cp, v_in, v_r, v_out)

    power = power_calc(wind, radius, cp, v_in, v_r, v_out, p_max)

    try:
        power =  power_calc(wind.tolist(), radius, cp, v_in, v_r, v_out)
    except AssertionError:
        assert True
    else:
        assert False

    wind_nan = wind
    wind_nan[0] = np.nan

    try:
        power =  power_calc(wind_nan, radius, cp, v_in, v_r, v_out)
    except AssertionError:
        assert True
    else:
        assert False

    assert wind.shape == power.shape

def test_build_lp_cst():
    ''' Test of function build_lp_cst'''

    power = np.array([0,1,2,3])
    n = len(power)
    dt = 1.0
    p_min = 1.0
    p_min_vec = np.array([0,0,0,0])
    p_max = 4.0
    losses_batt = 0.0
    losses_h2 = 0.0
    rate_batt = 1.0
    rate_h2 = 1.0
    max_soc = 1.0
    max_h2 = 1.0

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = build_lp_cst(power, dt, p_min,
                                                              p_max, n,
                                                              losses_batt,
                                                              losses_h2)

    n_x = 7*n +4
    n_eq = 4*n+2
    n_ineq = 6*n+1

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

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = build_lp_cst(power, dt,
                                                              p_min_vec,
                                                              p_max, n,
                                                              losses_batt,
                                                              losses_h2)

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = build_lp_cst(power, dt,
                                                              p_min_vec,
                                                              p_max, n,
                                                              losses_batt,
                                                              losses_h2,
                                                              rate_batt,
                                                              rate_h2,
                                                              max_soc,
                                                              max_h2)

def test_build_lp_obj_pareto():
    ''' Test of function build_lp_obj_pareto'''

    power = np.array([0,1,2,3])
    n = len(power)

    price = np.array([1, 1, 1, 1])
    eta = 0.5
    alpha = 0.5

    vec_obj = build_lp_obj_pareto(power, price,  n, eta, alpha)

    n_x = 7*n+6

    assert vec_obj.ndim == 1
    assert vec_obj.shape[0] == n_x

    try:
        vec_obj = build_lp_obj_pareto(power, price*0,  n, eta, alpha)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_lp_obj_pareto(power[:n-1], price,  n, eta, alpha)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_lp_obj_pareto(power, price[:n-1],  n, eta, alpha)
    except AssertionError:
        assert True
    else:
        assert False

def test_build_lp_obj_npv():
    ''' Test of function build_lp_obj_npv'''

    power = np.array([0,1,2,3])
    n = len(power)

    price = np.array([1, 1, 1, 1])
    discount_rate = 0.3
    n_year = 20

    vec_obj = build_lp_obj_npv(price, n, 1,1,1,1, discount_rate, n_year)

    n_x = 7*n+6

    assert vec_obj.ndim == 1
    assert vec_obj.shape[0] == n_x

    try:
        vec_obj = build_lp_obj_npv(price,  n, 1,1,1,1, 1.2, n_year)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_lp_obj_npv(price,  n, 1,1,1,1, discount_rate, -1)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_lp_obj_npv(price[:n-1], n,  1,1,1,1, discount_rate,
                                   n_year)
    except AssertionError:
        assert True
    else:
        assert False

def test_build_milp_obj():
    ''' Test of function build_milp_obj'''

    power = np.array([0,1,2,3])
    n = len(power)

    price = np.array([1, 1, 1, 1])
    eta = 0.5
    alpha = 0.5

    vec_obj = build_milp_obj(power, price,  n, eta, alpha)

    n_x = 14*n+7

    assert vec_obj.ndim == 1
    assert vec_obj.shape[0] == n_x

    try:
        vec_obj = build_milp_obj(power, price*0,  n, eta, alpha)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_milp_obj(power[:n-1], price,  n, eta, alpha)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        vec_obj = build_milp_obj(power, price[:n-1],  n, eta, alpha)
    except AssertionError:
        assert True
    else:
        assert False

def test_build_lp_cst_sparse():
    ''' Test of function build_lp_cst_sparse'''

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
        build_lp_cst_sparse(power, dt, p_min, p_max, n, losses_batt, losses_h2)
    n_x = 7*n +6
    n_eq = 2*n+2
    n_ineq = 10*n+2

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
        build_lp_cst_sparse(power, dt, p_min_vec, p_max, n, losses_batt,
                            losses_h2)

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse(power, dt, p_min_vec, p_max, n, losses_batt,
                            losses_h2, rate_batt, rate_h2, max_soc, max_h2)

    try:
        mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
            build_lp_cst_sparse(power[:n-1], dt, p_min_vec, p_max, n,
                                losses_batt, losses_h2)
    except AssertionError:
        assert True
    else:
        assert False

def test_build_milp_cst_sparse():
    ''' Test of function build_milp_cst_sparse'''
    power = np.array([0,1,2, 0, 3])
    n = len(power)
    dt = 1.0
    p_min = 1.0
    p_min_vec = np.array([0,0,0,1,0])
    p_max = 4.0
    losses_batt = 0.0
    losses_h2 = 0.0
    losses_fc = 0.0
    rate_batt = 1.0
    rate_h2 = 1.0
    max_soc = 1.0
    max_h2 = 1.0

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub, vec_int = \
        build_milp_cst_sparse(power, dt, p_min, p_max, n, losses_batt,
                              losses_batt, losses_h2, losses_fc)
    n_x = 14*n+7
    n_eq = 3*n+2
    n_ineq = 25*n+2

    # print(n_eq)
    # print(mat_eq.shape)

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
    assert vec_int.shape[0] == n_x
    assert vec_int.ndim == 1

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub, vec_int = \
        build_milp_cst_sparse(power, dt, p_min_vec, p_max, n, losses_batt,
                               losses_batt, losses_h2, losses_fc)

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub, vec_int = \
        build_milp_cst_sparse(power, dt, p_min_vec, p_max, n, losses_batt,
                              losses_batt, losses_h2, losses_fc, rate_batt,
                              rate_h2, max_soc, max_h2)

    try:
        mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub, vec_int = \
            build_milp_cst_sparse(power[:n-1], dt, p_min_vec, p_max, n,
                                  losses_batt, losses_batt, losses_h2,
                                  losses_fc)
    except AssertionError:
        assert True
    else:
        assert False


def test_solve_lp():
    ''' Test of function solve_lp'''

    power = np.array([2,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_min = 1.0
    p_min_vec = np.array([1,1,1,1,1])
    p_max = 4.0

    power_ts = TimeSeries(power, dt)
    price_ts = TimeSeries(price, dt)
    stor_batt = Storage(1,1,1,1,1,1)
    stor_h2 = Storage(1,1,1,1,1,1)
    eta = 0.5
    alpha = 0.5

    _ = solve_lp(power_ts, price_ts, stor_batt, stor_h2, eta, alpha, p_min,
                  p_max, n)

    _ = solve_lp(power_ts, price_ts, stor_batt, stor_h2, eta, alpha,
                  p_min_vec, p_max, n)

    try:
        _ = solve_lp(power_ts, TimeSeries(price, 2*dt), stor_batt, stor_h2,
                     eta, alpha, p_min_vec, p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        _ = solve_lp(power_ts, price_ts, stor_batt, stor_h2, eta, alpha, p_min,
                  0, n)
    except RuntimeError:
        assert True
    else:
        assert False



def test_linprog_mosek():
    ''' Test of function linprog_mosek'''
    power = np.array([0,1,2, 0, 3])
    n = len(power)
    dt = 1.0
    p_min = 1.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_max = 4.0
    losses_batt = 0.0
    losses_h2 = 0.0
    discount_rate = 0.03
    n_year = 20

    n_x = 7*n +6
    n_eq = 2*n+2
    n_ineq = 10*n+2

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse(power, dt, p_min, p_max, n, losses_batt, losses_h2)
    vec_obj = build_lp_obj_npv(price, n, 1,1,1,1, discount_rate, n_year)

    x = linprog_mosek(n_x, n_eq, n_ineq, mat_eq, vec_eq, mat_ineq, vec_ineq,
                      vec_obj, lb, ub)

    assert len(x) == n_x

    try:
        x = linprog_mosek(n_x-1, n_eq, n_ineq, mat_eq, vec_eq, mat_ineq,
                          vec_ineq, vec_obj, lb, ub)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = linprog_mosek(n_x, n_eq-1, n_ineq, mat_eq, vec_eq, mat_ineq,
                          vec_ineq, vec_obj, lb, ub)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = linprog_mosek(n_x, n_eq, n_ineq-1, mat_eq, vec_eq, mat_ineq,
                          vec_ineq, vec_obj, lb, ub)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = linprog_mosek(n_x, n_eq, n_ineq, mat_eq.toarray(), vec_eq,
                          mat_ineq, vec_ineq, vec_obj, lb, ub)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = linprog_mosek(n_x, n_eq, n_ineq, mat_eq, vec_eq,
                          mat_ineq.toarray(), vec_ineq, vec_obj, lb, ub)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = linprog_mosek(n_x, n_eq, n_ineq, mat_eq, vec_eq, mat_ineq,
                          vec_ineq, vec_obj, ub, lb)
    except RuntimeError:
        assert True
    else:
        assert False



def test_milp_mosek():
    ''' Test of function milp_mosek'''

    power = np.array([1,1,2, 1, 3])
    n = len(power)
    dt = 1.0
    p_min = 0.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_max = 4.0
    losses_batt = 0.0
    losses_h2 = 0.0
    losses_fc = 0

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub, vec_int = \
        build_milp_cst_sparse(power, dt, p_min, p_max, n, losses_batt,
                              losses_batt, losses_h2, losses_fc)
    n_x = 14*n+7
    n_eq = 3*n+2
    n_ineq = 25*n+2

    vec_obj = build_milp_obj(power, price, n, 0.5, 0.5)


    x = milp_mosek(n_x, n_eq, n_ineq, mat_eq, vec_eq, mat_ineq, vec_ineq,
                      vec_obj, lb, ub, vec_int)

    assert len(x) == n_x

    try:
        x = milp_mosek(n_x-1, n_eq, n_ineq, mat_eq, vec_eq, mat_ineq, vec_ineq,
                      vec_obj, lb, ub, vec_int)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = milp_mosek(n_x, n_eq-1, n_ineq, mat_eq, vec_eq, mat_ineq,
                       vec_ineq, vec_obj, lb, ub, vec_int)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = milp_mosek(n_x, n_eq, n_ineq-1, mat_eq, vec_eq, mat_ineq, vec_ineq,
                      vec_obj, lb, ub, vec_int)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = milp_mosek(n_x, n_eq, n_ineq, mat_eq.toarray(), vec_eq, mat_ineq,
                       vec_ineq, vec_obj, lb, ub, vec_int)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = milp_mosek(n_x, n_eq, n_ineq, mat_eq, vec_eq, mat_ineq.toarray(),
                       vec_ineq, vec_obj, lb, ub, vec_int)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        x = milp_mosek(n_x, n_eq, n_ineq, mat_eq, vec_eq, mat_ineq, vec_ineq,
                      vec_obj, ub, lb, vec_int)
    except RuntimeError:
        assert True
    else:
        assert False



def test_solve_lp_sparse_pareto():
    ''' Test of function solve_lp_sparse_pareto'''

    power = np.array([2,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_min = 0.5
    p_min_vec = np.array([0.5,0.5,0.5,0.5,0.5])
    p_max = 4.0

    power_ts = TimeSeries(power, dt)
    price_ts = TimeSeries(price, dt)
    stor_batt = Storage(1,1,1,1,1,1)
    stor_h2 = Storage(1,1,1,1,1,1)
    eta = 0.5
    alpha = 0.5

    _ = solve_lp_sparse_pareto(power_ts, price_ts, stor_batt, stor_h2, eta,
                               alpha, p_min, p_max, n)

    _ = solve_lp_sparse_pareto(power_ts, price_ts, stor_batt, stor_h2, eta,
                               alpha, p_min_vec, p_max, n)

    try:
        _ = solve_lp_sparse_pareto(power_ts, TimeSeries(price, 2*dt),
                                   stor_batt, stor_h2, eta, alpha,
                                   p_min_vec, p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        _ = solve_lp_sparse_pareto(power_ts, price_ts, stor_batt, stor_h2,
                                    eta, alpha, p_min, 0, n)
    except RuntimeError:
        assert True
    else:
        assert False



def test_solve_lp_sparse_old():
    ''' Test of function solve_lp_sparse_old'''
    power = np.array([2,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_min = 0.5
    p_min_vec = np.array([0.5,0.5,0.5,0.5,0.5])
    p_max = 4.0

    power_ts = TimeSeries(power, dt)
    price_ts = TimeSeries(price, dt)
    stor_batt = Storage(1,1,1,1,1,1)
    stor_h2 = Storage(1,1,1,1,1,1)
    discount_rate = 0.03
    n_year = 20

    _ = solve_lp_sparse_old(power_ts, price_ts, stor_batt, stor_h2,
                            discount_rate, n_year, p_min, p_max, n)

    _ = solve_lp_sparse_old(power_ts, price_ts, stor_batt, stor_h2,
                            discount_rate, n_year, p_min_vec, p_max, n)

    try:
        _ = solve_lp_sparse_old(power_ts, TimeSeries(price, 2*dt), stor_batt,
                                stor_h2, discount_rate, n_year, p_min_vec,
                                p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        _ = solve_lp_sparse_old(power_ts, price_ts, stor_batt, stor_h2,
                                    discount_rate, n_year, p_min, 0, n)
    except RuntimeError:
        assert True
    else:
        assert False



def test_solve_lp_sparse():
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

    _ = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min, p_max, n)

    _ = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min_vec, p_max, n)

    try:
        _ = solve_lp_sparse(TimeSeries(price, 2*dt), prod_wind, prod_pv,
                            stor_batt, stor_h2, discount_rate, n_year,
                            p_min_vec, p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        _ = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                                    discount_rate, n_year, p_min, 0, n)
    except RuntimeError:
        assert True
    else:
        assert False



def test_solve_milp_sparse():
    '''Test of function solve_milp_sparse'''
    power = np.array([2,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_min = 0.5
    p_min_vec = np.array([0.5,0.5,0.5,0.5,0.5])
    p_max = 4.0

    power_ts = TimeSeries(power, dt)
    price_ts = TimeSeries(price, dt)
    stor_batt = Storage(1,1,1,1,1,1)
    stor_h2 = Storage(1,1,1,1,1,1)
    eta = 0.5
    alpha = 0.5

    _ = solve_milp_sparse(power_ts, price_ts, stor_batt, stor_h2, eta,
                           alpha, p_min, p_max, n)

    _ = solve_milp_sparse(power_ts, price_ts, stor_batt, stor_h2, eta, alpha,
                           p_min_vec, p_max, n)

    try:
        _ = solve_milp_sparse(power_ts, TimeSeries(price, 2*dt), stor_batt,
                               stor_h2, eta, alpha, p_min_vec, p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        _ = solve_milp_sparse(power_ts, price_ts, stor_batt, stor_h2,
                                    eta, alpha, p_min, 0, n)
    except RuntimeError:
        assert True
    else:
        assert False
