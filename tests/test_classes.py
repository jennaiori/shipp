'''Module containing unit tests for the package classes'''

from shipp.components import Storage, Production, OpSchedule
from shipp.timeseries import TimeSeries

import numpy as np

def test_simple():
    '''
        Simple initialization test for the classes Storage, Production, OpSchedule and 
        TimeSeries
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

    stor_unit = Storage()
    stor_p = TimeSeries([1,1,1,1], dt)
    stor_e = TimeSeries([0, 1, 2, 3], dt)

    os = OpSchedule( [prod_unit], [stor_unit], [prod_unit.power], [stor_p], 
                    [stor_e])
    print(os)


def test_storage_initialization():
    # Test with all parameters
    stor = Storage(e_cap=100, p_cap=50, eff_in=0.9, eff_out=0.85, 
                   e_cost=1000, p_cost=5000, dod=0.8)
    
    # Test with default parameters
    stor_default = Storage()
    
    # Test with None values for capacities (used in optimization)
    stor_none = Storage(e_cap=None, p_cap=None)

def test_storage_methods():
    # Efficiency calculations
    stor = Storage(eff_in=0.9, eff_out=0.85)
    expected = 0.5 * (0.9 + 0.85)
    assert stor.get_av_eff() == expected

    expected = 0.9 * 0.85
    assert stor.get_rt_eff() == expected

    stor_asym = Storage(eff_in=0.95, eff_out=0.80)
    assert stor_asym.get_av_eff() != stor_asym.get_rt_eff()
    
    ## Test edge cases
    stor_perfect = Storage(eff_in=1.0, eff_out=1.0)
    assert stor_perfect.get_rt_eff() == 1.0
    
    stor_zero = Storage(eff_in=0, eff_out=1.0)
    assert stor_zero.get_rt_eff() == 0.0

    # Cost calculations
    stor = Storage(e_cap=100, p_cap=50, e_cost=1000, p_cost=5000)
    expected = 50 * 5000 + 100 * 1000
    assert stor.get_tot_costs() == expected
    
    ## Test with zero costs
    stor_free = Storage(e_cap=100, p_cap=50, e_cost=0, p_cost=0)
    assert stor_free.get_tot_costs() == 0
    
    ## Test with zero capacities
    stor_zero_cap = Storage(e_cap=0, p_cap=0, e_cost=1000, p_cost=5000)
    assert stor_zero_cap.get_tot_costs() == 0
    
    ## Test with None capacities (should handle gracefully or raise error)
    stor_none = Storage(e_cap=None, p_cap=None, e_cost=1000, p_cost=5000)
    try:
        stor_none.get_tot_costs()
    except ValueError:
        assert True
    else:
        assert False
    
    # Minimum energy level
    stor = Storage(e_cap=100, p_cap=50, dod = 0.8)
    expected_min_e = (1-0.8)*100
    assert stor.get_min_e() == expected_min_e

    # Check the behavior of the copy
    stor_copy = stor.copy()
    assert stor_copy.e_cap == stor.e_cap
    assert stor_copy.p_cap == stor.p_cap
    assert stor_copy.e_cost == stor.e_cost
    assert stor_copy.p_cost == stor.p_cost
    assert stor_copy.eff_in == stor.eff_in
    assert stor_copy.eff_out == stor.eff_out
    assert stor_copy.dod == stor.dod

    stor_copy.e_cap = 200
    assert stor_copy.e_cap != stor.e_cap


def test_opschedule_methods(): 

    # Initialization
    data = [1, 1, 1, 1]
    dt = 1
    ts = TimeSeries(data, dt)
    ts_empty =TimeSeries()

    prod_unit = Production(ts, p_cost = 8*1e3)

    stor_unit = Storage(e_cap = 10, p_cap = 5, e_cost = 500, p_cost = 600)
    stor_p = TimeSeries([1,1,1,1], dt)
    stor_e = TimeSeries([0, 1, 2, 3], dt)

    price = np.array([1, 0.5, 1, 1.5])
 
    # Test initialization without price
    os = OpSchedule( [prod_unit], [stor_unit], [prod_unit.power], [stor_p], 
                    [stor_e])
    assert os.revenue is None
    assert os.annual_revenue is None
    assert os.revenue_storage is None
    assert os.annual_revenue_storage is None

    ## Test correct initialization
    os = OpSchedule( [prod_unit], [stor_unit], [prod_unit.power], [stor_p], 
                    [stor_e], price)
    expected_capex = prod_unit.get_tot_costs() + stor_unit.get_tot_costs()
    expected_revenues = sum( [d*(p + s) for d,p,s in zip(price, data, data)])*dt
    expected_revenue_storage = sum( [d*(s) for d,s in zip(price, data)])*dt
    
    expected_annual_revenues = 365*24/(4*dt) * expected_revenues
    expected_annual_revenue_storage = 365*24/(4*dt) * expected_revenue_storage
    
    assert os.capex == expected_capex
    assert os.revenue == expected_revenues
    assert os.revenue_storage == expected_revenue_storage
    assert os.annual_revenue == expected_annual_revenues
    assert os.annual_revenue_storage == expected_annual_revenue_storage

    # Test initialization with different time step
    dt = 0.25
    prod_unit.power.dt = dt
    stor_p.dt = dt
    stor_e.dt = dt

    os = OpSchedule( [prod_unit], [stor_unit], [prod_unit.power], [stor_p], 
                    [stor_e], price)
    expected_capex = prod_unit.get_tot_costs() + stor_unit.get_tot_costs()
    expected_revenues = sum( [d*(p + s) for d,p,s in zip(price, data, data)])*dt
    expected_revenue_storage = sum( [d*(s) for d,s in zip(price, data)])*dt
    
    expected_annual_revenues = 365*24/(4*dt) * expected_revenues
    expected_annual_revenue_storage = 365*24/(4*dt) * expected_revenue_storage
    
    assert os.capex == expected_capex
    assert os.revenue == expected_revenues
    assert os.revenue_storage == expected_revenue_storage
    assert os.annual_revenue == expected_annual_revenues
    assert os.annual_revenue_storage == expected_annual_revenue_storage

    ## Test incorrect inputs - One input is not a list
    try:
        os = OpSchedule( prod_unit, [stor_unit], [prod_unit.power], [stor_p], 
                    [stor_e])
    except AssertionError:
         assert True
    else:
        assert False
    ## Test incorrect inputs - Difference in length of production objects and timeseries
    try:
        os = OpSchedule( [prod_unit, prod_unit], [stor_unit], [prod_unit.power], [stor_p], 
                    [stor_e])
    except AssertionError:
         assert True
    else:
        assert False
    ## Test incorrect inputs - Difference in length of storage objects and timeseris
    try:
        os = OpSchedule( [prod_unit], [stor_unit], [prod_unit.power], [stor_p, stor_p], 
                    [stor_e])
    except AssertionError:
         assert True
    else:
        assert False
    

    # Test revenue calculation with incorrect input
    try: 
        os.update_revenue(None)
    except AssertionError:
         assert True
    else:
        assert False
    
    # Test function for NPV and IRR calculation
    discount_rate = 0
    n_year = 2

    cash_flow = [-os.capex, os.annual_revenue]
    expected_irr = (os.annual_revenue/os.capex - 1)
    os.get_npv_irr(discount_rate, n_year)
    assert os.npv == sum(cash_flow)*1e-6
    assert os.irr == expected_irr
  
    # Test function for added NPV
    discount_rate = 0
    n_year = 2

    cash_flow = [-stor_unit.get_tot_costs(), os.annual_revenue_storage]
    expected_irr = (os.annual_revenue_storage/stor_unit.get_tot_costs() - 1)
    os.get_added_npv(discount_rate, n_year)
    assert os.a_npv == sum(cash_flow)*1e-6

    

    
    
