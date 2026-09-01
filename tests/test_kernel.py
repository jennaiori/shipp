'''Module containing unit tests for the module kernel.py'''

import numpy as np
import numpy_financial as npf
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel import build_lp_obj_npv, build_lp_cst_sparse, solve_lp_sparse, financial_metrics


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
    n_eq = 2*n
    n_ineq = 11*n+6

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
    n_eq = 0
    n_ineq = 15*n+6

    if mat_eq is not None:
        assert mat_eq.shape[0] == n_eq
        assert mat_eq.shape[1] == n_x
    if vec_eq is not None:
        assert vec_eq.shape[0] == n_eq
        assert mat_ineq.shape[0] == n_ineq
        assert vec_eq.ndim == 1
    assert mat_ineq.shape[1] == n_x
    assert vec_ineq.shape[0] == n_ineq
    assert lb.shape[0] == n_x
    assert ub.shape[0] == n_x
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

    # Check energy balance
    ops = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                        discount_rate, n_year, p_min, p_max, n)
    
    energy_in = dt*(sum(prod_wind.power.data) + sum(prod_pv.power.data))
    energy_delivered= dt*(sum(ops.power_out.data))
    energy_lost = dt*(sum(ops.losses[0])+ sum(ops.losses[1]))
    assert energy_in == energy_delivered + energy_lost

    # Check energy balance in case of curtailment
    
    ops = solve_lp_sparse(price_ts, prod_wind, prod_pv, stor1, stor2,
                        discount_rate, n_year, 0, 1.5, n)
    
    energy_in = dt*(sum(prod_wind.power.data) + sum(prod_pv.power.data))
    energy_delivered= dt*(sum(ops.power_out.data))
    energy_lost = dt*(sum(ops.losses[0])+ sum(ops.losses[1]))
    assert energy_in >= energy_delivered + energy_lost
    
    p_curtail = prod_wind.power.data - ops.production_p[0].data

    assert energy_in == energy_delivered + energy_lost + dt*sum(p_curtail)


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
        

def test_financial_metrics():
    '''Test of function financial_metrics'''
    power = np.array([1])
    dt = 1.0
    price = [1]
    price_ts = TimeSeries(price, dt)
    p_max_hpp = 1

    power_ts = TimeSeries(power, dt)
    power_half_ts = TimeSeries(0.5 * power, dt)
    zero_ts = TimeSeries(0 * power, dt)
    price_ts = TimeSeries(price, dt)

    stor1 = Storage(1, 1, 1, 1, 1, 1)
    stor_null = Storage(0, 0, 0, 0, 0, 0)

    discount_rate = 0.03
    m = 20
    p_cost_shared = 1

    prod1 = Production(power_ts, p_max = 3, p_cost=1)
    prod_null = Production(zero_ts, p_max = 0, p_cost=0)

    prod_list = [prod1, prod_null]
    prod_p = [power_ts, zero_ts]
    stor_list = [stor1, stor_null]
    stor_p = [power_half_ts, zero_ts]

    metrics = None
    for i, call in enumerate([
        # Check incorrect input
        lambda: financial_metrics(stor_list,prod_list, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        lambda: financial_metrics(prod_list, stor_list, [power, power], stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        lambda: financial_metrics( prod_list, stor_list, prod_p, [power, power], p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        lambda: financial_metrics( prod1, stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, [price], m, discount_rate),
        lambda: financial_metrics( prod_list, stor1, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate),  

        # Check mismatch in length
        lambda: financial_metrics(prod_list[:1], stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        lambda: financial_metrics(prod_list, stor_list[:1], prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        lambda: financial_metrics(prod_list, stor_list, prod_p[:1], stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        lambda: financial_metrics(prod_list, stor_list, prod_p, stor_p[:1], p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        # Incorrect values for the parameters
        lambda: financial_metrics(prod_list, stor_list, prod_p, stor_p, -p_max_hpp, p_cost_shared, price_ts, m, discount_rate),
        lambda: financial_metrics(prod_list, stor_list, prod_p, stor_p, p_max_hpp, -1, price_ts, m, discount_rate),
        lambda: financial_metrics(prod_list, stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, 0.1, discount_rate),
        lambda: financial_metrics(prod_list, stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, -10, discount_rate),
        lambda: financial_metrics(prod_list, stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, m, 2),
        lambda: financial_metrics(prod_list, stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, m, -0.1),

    ]):
        try:
            metrics = call()
   
        except AssertionError:
            assert True
        else:
            assert False, print(i)

    metrics = financial_metrics(prod_list, stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate)

    assert isinstance(metrics, (list, tuple))
    assert len(metrics) == 5

    # Check expected value
    lcoe, npv, irr, capex_tot, cashflow = metrics
    
    assert isinstance(cashflow, list) and len(cashflow) == m+1
    assert (0<= irr <=1)
    assert capex_tot >=0
    

    expected_capex_year0 = p_max_hpp*p_cost_shared + prod1.get_tot_costs() + stor1.get_tot_costs()
    expected_capex = p_max_hpp*p_cost_shared + prod1.get_tot_costs()
    for i in range(m//stor1.lifetime): 
        expected_capex+= stor1.get_tot_costs()/(1+discount_rate)**(i*stor1.lifetime)

    expected_revenues = np.dot(price, [p1+p2 for p1, p2 in zip(power_ts.data, power_half_ts.data)])

    assert expected_capex == capex_tot
    assert -expected_capex_year0 == cashflow[0]
    assert -stor1.get_tot_costs() + expected_revenues == cashflow[stor1.lifetime]
    assert all(expected_revenues == cashflow[i] for i in range(1, m+1) if i not in [i*stor1.lifetime for i in range(m//stor1.lifetime)] )

    expected_npv = npf.npv(discount_rate, cashflow)
    expected_irr = npf.irr(cashflow)

    cost_vec = np.zeros(m+1)
    cost_vec[0] = expected_capex_year0
    for i in range(1,m//stor1.lifetime): 
        cost_vec[i*stor1.lifetime] = stor1.get_tot_costs()
    aep_vec = [0] + [ sum(power_ts.data) + sum(power_half_ts.data) for _ in range(1,m+1)]
    expected_lcoe = npf.npv(discount_rate, cost_vec)/ npf.npv(discount_rate, aep_vec)

    assert expected_npv == npv
    assert expected_irr == irr
    assert expected_lcoe == lcoe

    # Check the values are unchanges if the inputs are swapped
    lcoe, npv, irr, capex_tot, cashflow = financial_metrics([prod_null, prod1], stor_list, [zero_ts, power_ts], stor_p, p_max_hpp, p_cost_shared, price_ts, m, discount_rate)
        
    assert expected_capex == capex_tot
    assert -expected_capex_year0 == cashflow[0]
    assert -stor1.get_tot_costs() + expected_revenues == cashflow[stor1.lifetime]
    assert all(expected_revenues == cashflow[i] for i in range(1, m+1) if i not in [i*stor1.lifetime for i in range(m//stor1.lifetime)] )
    assert expected_npv == npv
    assert expected_irr == irr
    assert expected_lcoe == lcoe
    
    lcoe, npv, irr, capex_tot, cashflow = financial_metrics(prod_list, [stor_null, stor1], prod_p, [zero_ts, power_half_ts], p_max_hpp, p_cost_shared, price_ts, m, discount_rate)
        
    assert expected_capex == capex_tot
    assert -expected_capex_year0 == cashflow[0]
    assert -stor1.get_tot_costs() + expected_revenues == cashflow[stor1.lifetime]
    assert all(expected_revenues == cashflow[i] for i in range(1, m+1) if i not in [i*stor1.lifetime for i in range(m//stor1.lifetime)] )
    assert expected_npv == npv
    assert expected_irr == irr
    assert expected_lcoe == lcoe

    # Check the behavior of added_price
    lcoe, npv, irr, capex_tot, cashflow = financial_metrics(prod_list, stor_list, prod_p, stor_p, p_max_hpp, p_cost_shared, zero_ts, m, discount_rate, added_price=1)
        
    assert expected_capex == capex_tot
    assert -expected_capex_year0 == cashflow[0]
    assert -stor1.get_tot_costs() + expected_revenues == cashflow[stor1.lifetime]
    assert all(expected_revenues == cashflow[i] for i in range(1, m+1) if i not in [i*stor1.lifetime for i in range(m//stor1.lifetime)] )
    assert expected_npv == npv
    assert expected_irr == irr
    assert expected_lcoe == lcoe

    # Check the correct implementation of OPEX

    discount_rate = 0
    stor1 = Storage(1, 1, 1, 1, 1, 1, opex_fix = 1, opex_var = 2)
    prod1 = Production(power_ts, p_max = 3, p_cost=1)

    lcoe, npv, irr, capex_tot, cashflow = financial_metrics([prod1, prod_null], [stor1, stor_null], prod_p, stor_p, p_max_hpp, p_cost_shared, zero_ts, m, discount_rate)

    expected_revenues = -stor1.opex_fix * stor1.p_cap - stor1.opex_var*sum(power_half_ts.data)*dt
    assert all(expected_revenues == cashflow[i] for i in range(1, (m//stor1.lifetime) * stor1.lifetime) if i not in [i*stor1.lifetime for i in range(m//stor1.lifetime)] )
   
    # Check when storage charges only (not variable opex in this case)
    lcoe, npv, irr, capex_tot, cashflow = financial_metrics([prod1, prod_null], [stor1, stor_null], prod_p, [TimeSeries([-0.5], dt), zero_ts], p_max_hpp, p_cost_shared, zero_ts, m, discount_rate)

    expected_revenues = -stor1.opex_fix * stor1.p_cap
    assert all(expected_revenues == cashflow[i] for i in range(1, (m//stor1.lifetime) * stor1.lifetime) if i not in [i*stor1.lifetime for i in range(m//stor1.lifetime)] )

    discount_rate = 0
    stor1 = Storage(1, 1, 1, 1, 1, 1)
    prod1 = Production(power_ts, p_max = 3, p_cost=1, opex_fix = 1, opex_var = 2)

    lcoe, npv, irr, capex_tot, cashflow = financial_metrics([prod1, prod_null], [stor1, stor_null], prod_p, stor_p, p_max_hpp, p_cost_shared, zero_ts, m, discount_rate)

    expected_revenues = -prod1.opex_fix * prod1.p_max - prod1.opex_var*sum(power_ts.data)*dt
    assert all(expected_revenues == cashflow[i] for i in range(1, (m//stor1.lifetime) * stor1.lifetime) if i not in [i*stor1.lifetime for i in range(m//stor1.lifetime)] )