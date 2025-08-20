'''Module containing unit tests for the module kernel.py'''

import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo, run_storage_operation, solve_dispatch_pyomo

def test_solve_lp_pyomo():
    power = np.array([2,1,2, 2, 3])
    n = len(power)
    dt = 1.0
    price = np.array([0.1, 0.1, 0.2, 0.1, 0.1])
    p_min = 0.5
    p_min_vec = np.array([0.5,0.5,0.5,0.5,0.5])
    p_max = 4.0
    dp_lim = 0.5

    power_ts = TimeSeries(0.5*power, dt)
    price_ts = TimeSeries(price, dt)
    stor_batt = Storage(1,1,1,1,1,1)
    stor_h2 = Storage(1,1,1,1,1,1)
    discount_rate = 0.03
    n_year = 20

    prod_wind = Production(power_ts, p_cost = 1)
    prod_pv = Production(power_ts, p_cost = 1)

    # Test the function with p_min as a scalar
    _ = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min, p_max, n)

    # Test the function with p_min as a vector
    _ = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min_vec, p_max, n)

    # Raise an error if the length of p_min_vec is not equal to the number of time steps
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

    # Test if the capacity changes when the input fixed_cap is True
    os = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, p_min, p_max, n, fixed_cap=True)
    assert os.storage_list[0].p_cap == 1
    assert os.storage_list[1].p_cap == 1
    assert os.storage_list[0].e_cap == 1
    assert os.storage_list[1].e_cap == 1

    # Test the function with the parameter dp_lim
    _ = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, 0, p_max, n, dp_lim = dp_lim)    
    # Test the function with an incorrect value for dp_lim
    try:
        _ = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, 0, p_max, n, dp_lim = -0.5)    

    except AssertionError:
        assert True
    else:
        assert False

    # Check that curtailment is correctly implemented
    p_max_curt = 2
    os = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, stor_h2,
                        discount_rate, n_year, 0, p_max_curt, n, fixed_cap=True)
    assert( max(os.power_out.data) <= p_max_curt)

    # Check that the depth of discharge is correctly implemented

    tol = 1e-5
    stor_batt_dod = Storage(1,1,1,1,1,1, dod = 0.9)
    stor_null = Storage(e_cap = 0, p_cap = 0)

    os = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt_dod, stor_null,
                        discount_rate, n_year, p_min, p_max, n)
    assert min(os.storage_e[0].data) >= stor_batt_dod.e_cap*(1 - stor_batt_dod.dod)-tol
    assert os.storage_list[0].dod ==0.9

    os = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_null, stor_batt_dod, 
                        discount_rate, n_year, p_min, p_max, n)
    assert min(os.storage_e[1].data) >= stor_batt_dod.e_cap*(1 - stor_batt_dod.dod)-tol
    assert os.storage_list[1].dod == 0.9


def test_run_storage_operation():
    power = [2, 1, 2, 2, 3, 5]
    price = [0.1, 0.1, 0.2, 0.1, 0.1, 0.3]
    p_min = 0.5
    p_max = 4.0
    e_start = 1.0
    n = 3
    nt = len(power) - n
    dt = 1.0
    stor = Storage(e_cap=2, p_cap=1, eff_in=0.9, eff_out=0.9, p_cost=1, e_cost=1)
    rel = 1.0
    dp_lim = 0.5

    # Test with default parameters
    ## Baseload
    result = run_storage_operation(
        run_type="unlimited",
        power=power,
        price=price,
        p_min=p_min,
        p_max=p_max,
        stor=stor,
        e_start=e_start,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
    )
    assert isinstance(result, dict)
    assert "power" in result
    assert "energy" in result
    assert "reliability" in result
    assert "revenues" in result
    assert "bin" in result

    ## Ramp-limitation
    result = run_storage_operation(
        run_type="unlimited",
        power=power,
        price=price,
        p_min=0,
        p_max=p_max,
        stor=stor,
        e_start=e_start,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
        dp_lim=dp_lim,
    )
    assert isinstance(result, dict)
    assert "power" in result
    assert "energy" in result
    assert "reliability" in result
    assert "revenues" in result
    assert "bin" in result

    # Test with forecast provided
    forecast = [[[2, 2, 2]], [[2, 2, 2]], [[2, 2, 2]]]
    ## Baseload
    result_with_forecast = run_storage_operation(
        run_type="forecast",
        power=power,
        price=price,
        p_min=p_min,
        p_max=p_max,
        stor=stor,
        e_start=e_start,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
        forecast=forecast,
    )
    assert isinstance(result_with_forecast, dict)
    assert "power" in result
    assert "energy" in result
    assert "reliability" in result
    assert "revenues" in result
    assert "bin" in result

    ## Ramp-limitation
    result_with_forecast = run_storage_operation(
        run_type="forecast",
        power=power,
        price=price,
        p_min=0,
        p_max=p_max,
        stor=stor,
        e_start=e_start,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
        forecast=forecast,
        dp_lim = dp_lim,
    )
    assert isinstance(result_with_forecast, dict)
    assert "power" in result
    assert "energy" in result
    assert "reliability" in result
    assert "revenues" in result
    assert "bin" in result

    # Test with invalid parameters (e.g., p_min > p_max)
    try:
        run_storage_operation(
            run_type="invalid_type",
            power=power,
            price=price,
            p_min=p_min,
            p_max=p_max,
            stor=stor,
            e_start=e_start,
            n=n,
            nt=nt,
            dt=dt,
            rel=rel,
        )
    except AssertionError:
        assert True
    else:
        assert False

    try:
        run_storage_operation(
            run_type="unlimited",
            power=power,
            price=price,
            p_min=p_min,
            p_max=p_max,
            stor=stor,
            e_start=e_start,
            n=n,
            nt=nt,
            dt=dt,
            rel=rel,
            dp_lim = -0.5
        )
    except AssertionError:
        assert True
    else:
        assert False

    try:
        run_storage_operation(
            run_type="rule-based",
            power=power,
            price=price,
            p_min=0,
            p_max=p_max,
            stor=stor,
            e_start=e_start,
            n=n,
            nt=nt,
            dt=dt,
            rel=rel,
            dp_lim = dp_lim
        )
    except RuntimeError:
        assert True
    else:
        assert False

    # Test with verbose mode enabled
    result_verbose = run_storage_operation(
        run_type="unlimited",
        power=power,
        price=price,
        p_min=p_min,
        p_max=p_max,
        stor=stor,
        e_start=e_start,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
        verbose=True,
    )
    assert isinstance(result_verbose, dict)
    assert "power" in result
    assert "energy" in result
    assert "reliability" in result
    assert "revenues" in result
    assert "bin" in result

    # Check that curtailment is correctly implemented:

    res = run_storage_operation(
        'unlimited',
         [100 for _ in range(24)],
         [1 for _ in range(24)],
         p_min = 0,
         p_max = 90,
         stor = stor,
         e_start = 0,
         n = 24,
         nt= 24,
         dt = 1)

    power_out = np.array([100 for _ in range(24)]) + np.array(res['power']) - np.array(res['p_cur'])

    assert( max(power_out) <= 90)

    # Check that the depth of discharge is correctly implemented
    tol = 1e-4
    stor_dod = Storage(e_cap=2, p_cap=1, eff_in=0.9, eff_out=0.9, p_cost=1, e_cost=1, dod = 0.9)
    e_start_dod = stor_dod.e_cap
    result = run_storage_operation(
        run_type="unlimited",
        power=power,
        price=price,
        p_min=p_min,
        p_max=p_max,
        stor=stor_dod,
        e_start=e_start_dod,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
    )

    assert min(result['energy']) >= stor_dod.e_cap*(1 - stor_dod.dod )- tol

    result = run_storage_operation(
        run_type="rule-based",
        power=power,
        price=price,
        p_min=p_min,
        p_max=p_max,
        stor=stor_dod,
        e_start=e_start_dod,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
    )

    assert min(result['energy']) >= stor_dod.e_cap*(1 - stor_dod.dod) - tol

    result = run_storage_operation(
        run_type="forecast",
        power=power,
        price=price,
        p_min=p_min,
        p_max=p_max,
        stor=stor_dod,
        e_start=e_start_dod,
        forecast=forecast,
        n=n,
        nt=nt,
        dt=dt,
        rel=rel,
    )

    assert min(result['energy']) >= stor_dod.e_cap*(1 - stor_dod.dod) - tol



def test_solve_dispatch_pyomo():
    power =[ [2,1,2, 2, 3]]
    dt = 1.0
    price = [0.1, 0.1, 0.2, 0.1, 0.1]
    p_min = 0.5
    p_max = 4.0
    dp_lim = 0.5

    stor_batt = Storage(1,1,1,1,1,1)
    stor_null = Storage(0,0)

    m = 1
    n = len(power[0])
    rel = 1.0
    e_start = 0

    # Test the output of the function
    p_vec1, e_vec1,  p_vec2, e_vec2, p_cur, bin, status = solve_dispatch_pyomo(price, m, rel, n, power, p_min, p_max, e_start, 0, dt,  stor_batt, stor_null)

    assert (len(p_vec1) == m)
    assert (len(p_vec2) == m)
    assert (len(e_vec1) == m)
    assert (len(e_vec2) == m)
    assert (len(p_cur) == m)
    assert (len(p_vec1_tmp) == n for p_vec1_tmp in p_vec1)
    assert (len(p_vec2_tmp) == n for p_vec2_tmp in p_vec2)
    assert (len(e_vec1_tmp) == n+1 for e_vec1_tmp in e_vec1)
    assert (len(e_vec2_tmp) == n+1 for e_vec2_tmp in e_vec2)
    assert (len(p_cur_tmp) == n for p_cur_tmp in p_cur)
    assert (len(bin) == n)

    assert(max(p_vec1_tmp) <= stor_batt.p_cap for p_vec1_tmp in p_vec1)
    assert(max(p_vec2_tmp) <= stor_null.p_cap for p_vec2_tmp in p_vec2)
    assert(max(e_vec1_tmp) <= stor_batt.e_cap for e_vec1_tmp in e_vec1)
    assert(max(e_vec2_tmp) <= stor_null.e_cap for e_vec2_tmp in e_vec2)
    
    assert(min(p_vec1_tmp) >= -stor_batt.p_cap for p_vec1_tmp in p_vec2)
    assert(min(p_vec2_tmp) >= -stor_null.p_cap for p_vec2_tmp in p_vec2)
    assert(min(e_vec1_tmp) >= 0 for e_vec1_tmp in e_vec1)
    assert(min(e_vec2_tmp) >= 0 for e_vec2_tmp in e_vec2)

    assert(e_vec1_tmp[0] == e_start for e_vec1_tmp in e_vec1)

    # Check that the function raises an error if the input is incorrect
    ## Incorrect format for the power forecast
    try:
        _ = solve_dispatch_pyomo(price, m, rel, n, power[0], p_min, p_max, e_start, 0, dt,  stor_batt, stor_null)
    except TypeError:
        assert True
    else:
        assert False

    ## Incorrect value for the reliability
    try:
        _ = solve_dispatch_pyomo(price, m, rel+2.0, n, power, p_min, p_max, e_start, 0, dt,  stor_batt, stor_null)
    except AssertionError:
        assert True
    else:
        assert False

    # Incorrect length for the price
    try:
        _ = solve_dispatch_pyomo(price[:n-2], m, rel+2.0, n, power, p_min, p_max, e_start, 0, dt,  stor_batt, stor_null)
    except AssertionError:
        assert True
    else:
        assert False

    # Negative price value
    price_neg = [0.1, 0.1, 0.2, -0.2, 0.1]
    try:
        _ = solve_dispatch_pyomo(price_neg, m, rel, n, power, p_min, p_max, e_start, 0, dt,  stor_batt, stor_null)
    except AssertionError:
        assert True
    else:
        assert False

    # Raise an error if the problem is ill-posed / not converged
    try:
        _ = solve_dispatch_pyomo(price, m, rel, n, power, p_min, -p_max, e_start, 0, dt,  stor_batt, stor_null)
    except RuntimeError:
        assert True
    else:
        assert False

    # Test the function with the parameter dp_lim
    p_vec1, e_vec1,  p_vec2, e_vec2, p_cur, bin, status = solve_dispatch_pyomo(price, m, rel, n, power, p_min, p_max, e_start, 0, dt,  stor_batt, stor_null, dp_lim = dp_lim)  
    # Test the function with a positive value for dp_lim
    try:
        _ = solve_dispatch_pyomo(price, m, rel, n, power, p_min, p_max, e_start, 0, dt,  stor_batt, stor_null, dp_lim = -dp_lim)  

    except AssertionError:
        assert True
    else:
        assert False


    # Check that the depth of discharge is correctly 

    stor_batt_dod = Storage(1,1,1,1,1,1, dod = 0.9)
    e_start_dod = 0.9
    p_vec1, e_vec1,  p_vec2, e_vec2, p_cur, bin, status = solve_dispatch_pyomo(price, m, rel, n, power, p_min, p_max, e_start_dod, 0, dt,  stor_batt_dod, stor_null)

    assert min(e_vec1[0]) >= stor_batt_dod.e_cap*(1 - stor_batt_dod.dod)
    
    p_vec1, e_vec1,  p_vec2, e_vec2, p_cur, bin, status = solve_dispatch_pyomo(price, m, rel, n, power, p_min, p_max,  0, e_start_dod, dt,   stor_null, stor_batt_dod)

    assert min(e_vec2[0]) >= stor_batt_dod.e_cap*(1 - stor_batt_dod.dod)

    
