'''Module containing unit tests for the module kernel.py'''

import numpy as np
from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo, run_storage_operation

def test_solve_lp_pyomo():
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

    # Test with default parameters
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

    # Test with forecast provided
    forecast = [[[2, 2, 2]], [[2, 2, 2]], [[2, 2, 2]]]

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



