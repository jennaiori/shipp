'''Module containing unit tests for the module kernel.py'''

import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel import build_lp_obj_npv, build_lp_cst_sparse, solve_lp_sparse


def test_build_lp_obj_npv_lp():
    ''' Test of function build_lp_obj_npv'''

    power = np.array([0,1,2,3])
    n = len(power)

    price = np.array([1, 1, 1, 1])
    discount_rate = 0.3
    n_year = 20

    vec_obj = build_lp_obj_npv(price, n, 1,1,1,1, discount_rate, n_year, options = dict(formulation = 'lp'))

    n_x = 7*n+6

    assert vec_obj.ndim == 1
    assert vec_obj.shape[0] == n_x

    try:
        vec_obj = build_lp_obj_npv(price,  n, 1,1,1,1, 1.2, n_year, options = dict(formulation = 'lp'))
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
                                   n_year, options = dict(formulation = 'lp'))
    except AssertionError:
        assert True
    else:
        assert False

def test_build_lp_obj_npv_lp_alt():
    ''' Test of function build_lp_obj_npv'''

    power = np.array([0,1,2,3])
    n = len(power)

    price = np.array([1, 1, 1, 1])
    discount_rate = 0.3
    n_year = 20

    vec_obj = build_lp_obj_npv(price, n, 1,1,1,1, discount_rate, n_year, options = dict(formulation = 'lp_alt'))

    n_x = 5*n+6

    assert vec_obj.ndim == 1
    assert vec_obj.shape[0] == n_x

    try:
        vec_obj = build_lp_obj_npv(price,  n, 1,1,1,1, 1.2, n_year, options = dict(formulation = 'lp_alt'))
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
                                   n_year, options = dict(formulation = 'lp_alt'))
    except AssertionError:
        assert True
    else:
        assert False


def test_build_lp_cst_sparse_lp():
    ''' Test of function build_lp_cst_sparse'''

    power = np.array([0,1,2, 0, 3])
    n = len(power)
    dt = 1.0
    p_min = 1.0
    p_min_vec = np.array([0,0,0,1,0])
    p_max = 4.0
    stor1 = Storage(1,1,1,1,1,1)
    stor2 = Storage(1,1,1,1,1,1)

    rate_batt = 1.0
    rate_h2 = 1.0
    max_soc = 1.0
    max_h2 = 1.0

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse(power, dt, p_min, p_max, n, stor1, stor2, options = dict(formulation = 'lp'))
    n_x = 7*n+6
    n_eq = 2 + 2*n
    n_ineq = 10*n+4

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
        build_lp_cst_sparse(power, dt, p_min_vec, p_max, n, stor1,
                            stor2, options = dict(formulation = 'lp'))

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse(power, dt, p_min_vec, p_max, n, stor1,
                            stor2, rate_batt, rate_h2, max_soc, max_h2, options = dict(formulation = 'lp'))

    try:
        mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
            build_lp_cst_sparse(power[:n-1], dt, p_min_vec, p_max, n,
                                stor1, stor2, options = dict(formulation = 'lp'))
    except AssertionError:
        assert True
    else:
        assert False

def test_build_lp_cst_sparse_lp_alt():
    ''' Test of function build_lp_cst_sparse'''

    power = np.array([0,1,2, 0, 3])
    n = len(power)
    dt = 1.0
    p_min = 1.0
    p_min_vec = np.array([0,0,0,1,0])
    p_max = 4.0
    stor1 = Storage(1,1,1,1,1,1)
    stor2 = Storage(1,1,1,1,1,1)

    rate_batt = 1.0
    rate_h2 = 1.0
    max_soc = 1.0
    max_h2 = 1.0

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse(power, dt, p_min, p_max, n, stor1, stor2, options = dict(formulation = 'lp_alt'))
    n_x = 5*n+6
    n_eq = 2
    n_ineq = 14*n+4

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
        build_lp_cst_sparse(power, dt, p_min_vec, p_max, n, stor1,
                            stor2, options = dict(formulation = 'lp_alt'))

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
        build_lp_cst_sparse(power, dt, p_min_vec, p_max, n, stor1,
                            stor2, rate_batt, rate_h2, max_soc, max_h2, options = dict(formulation = 'lp_alt'))

    try:
        mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = \
            build_lp_cst_sparse(power[:n-1], dt, p_min_vec, p_max, n,
                                stor1, stor2, options = dict(formulation = 'lp_alt'))
    except AssertionError:
        assert True
    else:
        assert False




def test_solve_lp_sparse():
    '''Test of function solve_lp_sparse'''
    power = np.array([2,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = [0.1, 0.1, 0.2, 0.1, 0.1]
    p_min = 0.5
    p_min_vec = np.array([0.5,0.5,0.5,0.0,0.5])
    p_max = 4.0

    assert isinstance(price, list)
    assert all(isinstance(p, (int, float)) for p in price)

    power_ts = TimeSeries(0.5*power, dt)
    price_ts = TimeSeries(price, dt)
    stor1 = Storage(1,1,1,1,1,1)
    stor2 = Storage(1,1,1,1,1,1)
    discount_rate = 0.03
    n_year = 20

    prod_wind = Production(power_ts, p_cost = 1)
    prod_pv = Production(power_ts, p_cost = 1)
    
    _ = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                        discount_rate, n_year, p_min, p_max, n)

    _ = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                        discount_rate, n_year, p_min_vec, p_max, n)

    try:
        _ = solve_lp_sparse(TimeSeries(price, 2*dt), prod_wind, prod_pv,
                            stor1, stor2, discount_rate, n_year,
                            p_min_vec, p_max, n)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        _ = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                                    discount_rate, n_year, 4.0, 0.0, n)
    except RuntimeError:
        assert True
    else:
        assert False


def test_solve_lp_sparse_formulation():
    '''Test of function solve_lp_sparse comparing the three possible formulations'''
    power = np.array([0.5,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = [0.05, 0.1, 0.2, 0.3, 0.35]
    p_min = 0.5
    p_max = 4.0

    assert isinstance(price, list)
    assert all(isinstance(p, (int, float)) for p in price)

    power_ts = TimeSeries(0.5*power, dt)
    price_ts = TimeSeries(price, dt)
    stor2 = Storage(0.5,0.5,1.0,1.0,1,1)
    stor1 = Storage(1.0,1.0,0.5,0.5,10,10)

    discount_rate = 0.03
    n_year = 20

    prod_wind = Production(power_ts, p_cost = 1)
    prod_pv = Production(power_ts, p_cost = 1)
    
    for p_min, p_max in zip([0, 0.1, 0.2, 0.2], [4.0, 4.0, 4.0, 2.5]):

        os1 = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                            discount_rate, n_year, p_min, p_max, n, options = dict(fixed_cap = True, formulation = 'lp', epsilon = 0.0))

        os2 = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                            discount_rate, n_year, p_min, p_max, n, options = dict(fixed_cap = True, formulation = 'lp_alt'))

        os3 = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                            discount_rate, n_year, p_min, p_max, n, options = dict(fixed_cap = True, formulation = 'milp', epsilon = 0.0))
  
        assert (os1.production_p[0].data == os2.production_p[0].data).all()
        assert (os1.production_p[1].data == os2.production_p[1].data).all()
        assert (os1.storage_p[0].data == os2.storage_p[0].data).all(), print(p_min, p_max, os1.storage_p[0].data, os2.storage_p[0].data)
        assert (os1.storage_p[1].data == os2.storage_p[1].data).all(), print(p_min, p_max,'\n os1\n', os1.storage_list[0], os1.storage_list[1], '\n', os1.storage_p[0].data, os1.storage_p[1].data,'\n', os1.power_out.data, '\n os2\n', os2.storage_list[0], os2.storage_list[1], '\n', os2.storage_p[0].data , os2.storage_p[1].data,'\n', os2.power_out.data)
        assert (os1.storage_e[0].data == os2.storage_e[0].data).all()
        assert (os1.storage_e[1].data == os2.storage_e[1].data).all()

        assert (os1.production_p[0].data == os3.production_p[0].data).all()
        assert (os1.production_p[1].data == os3.production_p[1].data).all()
        assert (os1.storage_p[0].data == os3.storage_p[0].data).all(), print(p_min, p_max, os1.storage_p[0].data, os3.storage_p[0].data)
        assert (os1.storage_p[1].data == os3.storage_p[1].data).all(), print(p_min, p_max,'\n os1\n', os1.storage_list[0], os1.storage_list[1], '\n', os1.storage_p[0].data, os1.storage_p[1].data,'\n', os1.power_out.data, '\n os3\n', os3.storage_list[0], os3.storage_list[1], '\n', os3.storage_p[0].data , os3.storage_p[1].data,'\n', os3.power_out.data)
        assert (os1.storage_e[0].data == os3.storage_e[0].data).all()
        assert (os1.storage_e[1].data == os3.storage_e[1].data).all()

        os1 = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                            discount_rate, n_year, p_min, p_max, n, options = dict(fixed_cap = False, formulation = 'lp', epsilon = 1e-6, alpha_obj = 1.0))

        os2 = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                            discount_rate, n_year, p_min, p_max, n, options = dict(fixed_cap = False, formulation = 'lp_alt', alpha_obj = 1.0))

        os3 = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                            discount_rate, n_year, p_min, p_max, n, options = dict(fixed_cap = False, formulation = 'milp', epsilon = 0.0, alpha_obj = 1.0))
  
        assert (os1.production_p[0].data == os2.production_p[0].data).all()
        assert (os1.production_p[1].data == os2.production_p[1].data).all()
        assert (os1.storage_list[0].p_cap == os2.storage_list[0].p_cap), print(os1.storage_list[0].p_cap, os2.storage_list[0].p_cap)
        assert (os1.storage_list[0].e_cap == os2.storage_list[0].e_cap), print(os1.storage_list[0].e_cap, os2.storage_list[0].e_cap)
        assert (os1.storage_list[1].p_cap == os2.storage_list[1].p_cap), print(os1.storage_list[0], '\n', os1.storage_list[1], '\n', os2.storage_list[0] ,'\n', os2.storage_list[1])
        assert (os1.storage_list[1].e_cap == os2.storage_list[1].e_cap), print(os1.storage_list[1].e_cap, os2.storage_list[1].e_cap)
        assert (os1.storage_p[0].data == os2.storage_p[0].data).all(), print(os1.storage_p[0].data, os2.storage_p[0].data)
        assert (os1.storage_p[1].data == os2.storage_p[1].data).all()
        assert (os1.storage_e[0].data == os2.storage_e[0].data).all()
        assert (os1.storage_e[1].data == os2.storage_e[1].data).all()

        assert (os1.production_p[0].data == os3.production_p[0].data).all()
        assert (os1.production_p[1].data == os3.production_p[1].data).all()
        assert (os1.storage_list[0].p_cap == os3.storage_list[0].p_cap), print(os1.storage_list[0].p_cap, os3.storage_list[0].p_cap)
        assert (os1.storage_list[0].e_cap == os3.storage_list[0].e_cap), print(os1.storage_list[0].e_cap, os3.storage_list[0].e_cap)
        assert (os1.storage_list[1].p_cap == os3.storage_list[1].p_cap), print(os1.storage_list[0], '\n', os1.storage_list[1], '\n', os3.storage_list[0] ,'\n', os3.storage_list[1])
        assert (os1.storage_list[1].e_cap == os3.storage_list[1].e_cap), print(os1.storage_list[1].e_cap, os3.storage_list[1].e_cap)
        assert (os1.storage_p[0].data == os3.storage_p[0].data).all(), print(os1.storage_p[0].data, os3.storage_p[0].data)
        assert (os1.storage_p[1].data == os3.storage_p[1].data).all()
        assert (os1.storage_e[0].data == os3.storage_e[0].data).all()
        assert (os1.storage_e[1].data == os3.storage_e[1].data).all()
        

