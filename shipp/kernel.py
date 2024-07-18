"""This module defines kernel functions for shipp.

The functions defined in this module are used to analyze or compute data
for the classes defined in shipp.

Functions:
    build_lp_obj_npv_sf: Build objective vector for NPV maximization.
    build_lp_cst_sparse_sf: Build sparse constraints for a LP.
    solve_lp_sparse_sf: Build and solve a LP for NPV maximization.
    os_rule_based: Build the operation schedule with a rule-based EMS
"""

import traceback
import numpy as np
import numpy_financial as npf
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import scipy.sparse as sps

from shipp.components import Storage, OpSchedule, Production
from shipp.timeseries import TimeSeries

def build_lp_obj_npv_sf(price: np.ndarray, n: int, batt_p_cost: float,
                     batt_e_cost: float, h2_p_cost: float, h2_e_cost: float,
                     discount_rate: float, n_year: int) -> np.ndarray:
    """Build objective vector for a linear program for NPV maximization.

    This function returns an objective vector corresponding the
    maximization of Net Present Value (NPV):

        f(x) = - factor*price*power + price_e_cap_batt*e_cap_batt
               + price_p_cap_batt*p_cap_batt
               + price_e_cap_h2*e_cap_h2 + price_p_cap_h2*p_cap_h2

    where factor = sum_n=1^(n_year) (1+discount_rate)**(-n)

    The objective vector corresponds to the following design variables:
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max battery power capacity, shape-(1,)
        Max state of charge (battery energy capacity), shape-(1,)
        Max hydrogen system power capacity, shape-(1,)
        Max hydrogen system energy capacity, shape-(1,)

    The number of design variables is n_x = 4*n+6

    Params:
        price (np.array): An array for the price of electricity to
            calculate the revenue [currency/MWh]
        n (int): the number of time steps [-]
        batt_p_cost (float): cost of battery for power capacity
            [currency/MW]
        batt_e_cost (float): cost of battery for energy capacity
            [currency/MWh]
        h2_p_cost (float): cost of hydrogen storage system for power
            capacity [currency/MW]
        h2_e_cost (float): cost of hydrogen for energy capacity
            [currency/MWh]
        discount_rate (float): discount rate to calculate the NPV [-]
        n_year (int): number of years of operation of the project [-]

    Returns:
        vec_obj (np.ndarray): A shape-(n_x,) array representing the
            objective function of the linear program [-]

    Raises:
        AssertionError if the length of the price is below n, if any input is not finite
    """

    assert len(price) >= n
    assert np.all(np.isfinite(price))
    assert n != 0
    assert np.isfinite(batt_p_cost)
    assert np.isfinite(batt_e_cost)
    assert np.isfinite(h2_p_cost)
    assert np.isfinite(h2_e_cost)
    assert n_year > 0
    assert 0 <= discount_rate <= 1


    factor = npf.npv(discount_rate, np.ones(n_year))-1

    normed_price = 365 * 24 / n * np.reshape(price[:n], (n,1))*factor

    vec_obj = np.vstack((-normed_price,             # Batt power
                        -normed_price,             # Power from fuel cell / to electrolyzer
                        np.zeros((n+1, 1)),
                        np.zeros((n+1, 1)),
                        batt_p_cost*np.ones((1,1)),           # minimize max batt power
                        batt_e_cost*np.ones((1,1)),           # minimize max state of charge
                        h2_p_cost*np.ones((1,1)),             # minimize max h2 power rate
                        h2_e_cost*np.ones((1,1)))).squeeze()  # minimize max h2 energy capacity

    return vec_obj

def build_lp_cst_sparse_sf(power: np.ndarray, dt: float, p_min: float|np.ndarray,
                        p_max: float, n: int, eps_batt: float,
                        eps_h2: float, rate_batt: float = -1.0,
                        rate_h2: float = -1.0, max_soc: float = -1.0,
                        max_h2: float = -1.0) -> tuple[sps.coo_matrix,
                                                       np.ndarray,
                                                       sps.coo_matrix,
                                                       np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray]:
    """Build sparse constraints for a LP.

    Function to build the matrices and vectors corresponding to the
    linear program representing the scheduling design problem. A sparse
    format is used to represent the matrices.

    The constraints are made of equality and inequality constraints such
    that:

        mat_eq * x = vec_eq
        mat_ineq * x <= vec_ineq
        bounds_lower <= x <= bounds_upper

    With n the number of time steps, the problem is made of:
        n_x = 4*n+6 design variables
        n_eq = 2 equality constraints
        n_ineq = 12*n+2 inequality constraints

    The design variables for the linear problem are:
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max battery power capacity, shape-(1,)
        Max state of charge (battery energy capacity), shape-(1,)
        Max hydrogen system power capacity, shape-(1,)
        Max hydrogen system energy capacity, shape-(1,)

    The equality constraints for the problem are:
        Constraint to enforce the value of the first state of charge
            is equal to the last (size 1)
        Constraint to enforce the value of the first hydrogen level
            is equal to the last (size 1)

    The inequality constraints for the problem are:
        Constraints on the minimum and maximum combined power from
            renewables and storage systems (size 2*n)
        Constraints on the battery state of charge (size 2*n)
        Constraints on the hydrogen storage levels (size 2*n)
        Constraints on maxmimum and minimum power to and from the
            hydrogen storage system (size 2*n)
        Constraints on the maxmimum and minimum power to and from the
            battery (size 2*n)
        Constraints on the maximum state of charge (size n+1)
        Constraints on the maximum hydrogen levels (size n+1)


    Params:
        power (np.ndarray): A shape-(n,) array for the power production
            from renewables [MW].
        dt (float): time step [s].
        p_min (float or np.ndarray): A float or shape-(n,) array for the
             minimum required power production [MW].
        p_max (float): maximum power production [MW]
        n (int): number of time steps [-].
        eps_batt (float): represents the portion of power lost during
            charge and discharge of the battery [-]. The same efficiency
            is assumed for charge and discharge.
        eps_h2 (float): represents the portion of power lost during
            charge and discharge of the hydrogen storage system [-]. The
            same efficiency is assumed for charge and discharge.
        rate_batt (float, optional): maximum power rate for the battery
            [MW]. Default to -1.0 when there is no limit in power rate.
        rate_h2 (float): maximum power rate for the hydrogen storage
            system [MW]. Default to -1.0 when there is no limit in power
             rate.
        max_soc (float): maximum state of charge for the battery [MWh].
            Default to -1.0 when there is no limit in state of charge.
        max_soc (float): maximum hydrogen level for the hydrogen storage
            [MWh]. Default to -1.0 when there is no limit in storage.

    Returns:
        mat_eq (sps.coo_matrix): A shape-(n_eq, n_x) array for the matrix
            of the equality constraints [-]
        vec_eq (np.ndarray): A shape-(n_eq,) array for the vector of the
             equality constraints [-]
        mat_ineq (sps.coo_matrix): A shape-(n_ineq, n_x) array for the
            matrix of the inequality constraints [-]
        vec_ineq (np.ndarray): A shape-(n_ineq,) array for the vector of
             the inequality constraints [-]
        bounds_lower (np.ndarray): A shape-(n_x,) array for the lower
            bounds [-]
        bounds_upper (np.ndarray): A shape-(n_x,) array for the upper
            bounds [-]

    Raises:
        ValueError: if argument p_min is not a float or a list of floats
        AssertionError: if any argument is not finite and if the
            argument power has a length lower than n
    """

    assert np.all(np.isfinite(power))
    assert len(power) >= n
    assert np.all(np.isfinite(p_min))
    assert np.isfinite(p_max)
    assert np.isfinite(dt)
    assert np.isfinite(eps_batt)
    assert np.isfinite(eps_h2)

    if rate_batt == -1 or rate_batt is None:
        rate_batt = p_max

    if rate_h2 == -1 or rate_h2 is None:
        rate_fc = p_max
        rate_h2 = p_max
    else:
        rate_fc = rate_h2

    if max_soc == -1 or max_soc is None:
        max_soc = n*dt*rate_batt #MWh

    if max_h2 == -1 or max_h2 is None:
        max_h2 = 100*1400*0.0333 #MWh  (1400kg H2 and 1kg is 33.3 kWh)

    assert np.isfinite(rate_batt)
    assert np.isfinite(rate_h2)
    assert np.isfinite(max_soc)
    assert np.isfinite(max_h2)

    if isinstance(p_min, (np.ndarray, list)):
        assert len(p_min) >= n
        p_min_vec = p_min[:n].reshape(n,1)
    elif isinstance(p_min, (float, int)):
        p_min_vec = p_min * np.ones((n,1))
    else:
        raise ValueError("Input p_min in build_lp_cost must be a float, int,\
                          list or numpy.array")

    z_n = sps.coo_array((n,n))
    z_np1 = sps.coo_array((n+1,n+1))
    z_n_np1 = sps.coo_array((n , n+1))
    z_np1_n = z_n_np1.transpose()
    z_1n = sps.coo_array((1,n))
    z_n1 = sps.coo_array((n, 1))
    z_11 = sps.coo_array((1 , 1))
    z_1_np1 = sps.coo_array((1 , n+1))
    z_np1_1 = z_1_np1.transpose()
    eye_n = sps.eye(n)
    eye_np1 = sps.eye(n+1)
    one_11 = sps.coo_array(np.ones((1  ,1)))
    one_n1 = sps.coo_array(np.ones((n,1)))
    one_np1_1 = sps.coo_array(np.ones((n+1, 1)))

    mat_last_soc = sps.vstack((sps.coo_array((n-1,1)), -1*one_11))
    mat_diag_soc = eye_n - sps.diags(np.ones(n-1),1)

    # upper bound on power
    power_ub =(np.array([ max(0, p_max - p) for p in  power[:n]])).reshape(n,1)



    # EQUALITY CONSTRAINTS
    # Constraint on first state of charge
    mat_first_soc = sps.hstack((z_1n, z_1n,
                            one_11, np.zeros((1, n-1)), -one_11,
                            z_1_np1,
                            z_11, z_11, z_11, z_11))
    vec_first_soc = z_11

    # Constraint on first h2 level
    mat_first_soc_h2 = sps.hstack((z_1n, z_1n,
                            z_1_np1,
                            one_11, np.zeros((1, n-1)), -one_11,
                            z_11, z_11, z_11, z_11))
    vec_first_soc_h2 = z_11

    # INEQ CONSTRAINT
    # Constraint on wind power + battery power + h2 power >= p_min
    mat_power_bound = sps.hstack((eye_n, eye_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    vec_power_min = p_min_vec - power[:n].reshape(n,1)
    vec_power_max = power_ub

    # Constraint on the maximum state of charge (i.e battery capacity)
    mat_max_soc = sps.hstack((z_np1_n, z_np1_n, eye_np1, z_np1,
                                z_np1_1, -1*one_np1_1, z_np1_1, z_np1_1))
    vec_max_soc = z_np1_1


    mat_max_batt = sps.hstack((eye_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
    vec_max_batt = z_n1

    mat_min_batt = sps.hstack(( -eye_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
    vec_min_batt = z_n1

    # Constraint on electrolyzer / fuel cell maximum power output
    mat_h2_max_power = sps.hstack((z_n, eye_n, z_n_np1, z_n_np1,
                                   z_n1, z_n1, -one_n1, z_n1))
    vec_h2_max_power = z_n1
    mat_h2_min_power = sps.hstack((z_n, -eye_n, z_n_np1, z_n_np1,
                                   z_n1, z_n1, -one_n1, z_n1))
    vec_h2_min_power = z_n1

    mat_max_h2 = sps.hstack((z_np1_n, z_np1_n, z_np1,  eye_np1,
                             z_np1_1, z_np1_1, z_np1_1, -1*one_np1_1,))
    vec_max_h2 = z_np1_1

    # Constraint on state of charge, including storage losses
    # soc_(n+1) - soc_(n) <= - dt * P_batt
    mat_soc_sts1 = sps.hstack((dt * eye_n, z_n,
                            -mat_diag_soc, -mat_last_soc, z_n_np1,
                            z_n1, z_n1, z_n1, z_n1))
    vec_soc_sts1 = z_n1

    # soc_(n+1) - soc_(n) <= - dt/eta * P_batt
    mat_soc_sts2 = sps.hstack((dt/(1-eps_batt) * eye_n, z_n,
                            -mat_diag_soc, -mat_last_soc, z_n_np1,
                            z_n1, z_n1, z_n1, z_n1))
    vec_soc_sts2 = z_n1

    # Constraint on hydrogen levels:
    #  soc_(n+1) - soc_(n) <= - dt * P_batt
    mat_soc_lts1 = sps.hstack((z_n, dt * eye_n,
                                z_n_np1, -mat_diag_soc, -mat_last_soc,
                                z_n1, z_n1, z_n1, z_n1))
    vec_soc_lts1 = z_n1

    #  soc_(n+1) - soc_(n) <= - dt/eta * P_batt
    mat_soc_lts2 = sps.hstack((z_n, dt/(1-eps_h2) * eye_n,
                                z_n_np1, -mat_diag_soc, -mat_last_soc,
                                z_n1, z_n1, z_n1, z_n1))
    vec_soc_lts2 = z_n1

    ## Assemble matrices
    mat_eq = sps.vstack((mat_first_soc,  mat_first_soc_h2))
    vec_eq = sps.vstack((vec_first_soc,  vec_first_soc_h2)).toarray().squeeze()

    mat_ineq = sps.vstack((-1*mat_power_bound, mat_power_bound,
                                mat_soc_sts1, mat_soc_sts2,
                                mat_soc_lts1, mat_soc_lts2,
                                mat_max_soc, mat_h2_max_power,
                                mat_h2_min_power, mat_max_batt,
                                mat_min_batt, mat_max_h2))
    vec_ineq = sps.vstack((-1*vec_power_min, vec_power_max,
                                vec_soc_sts1, vec_soc_sts2,
                                vec_soc_lts1, vec_soc_lts2,
                                vec_max_soc, vec_h2_max_power,
                                vec_h2_min_power, vec_max_batt,
                                vec_min_batt, vec_max_h2)).toarray().squeeze()
    # BOUNDS ON DESIGN VARIABLES
    bounds_lower = sps.vstack((-rate_batt * one_n1,
                                -rate_h2 * one_n1,
                                z_np1_1,
                                z_np1_1,
                                z_11,
                                z_11,
                                z_11,
                                z_11)).toarray().squeeze()

    bounds_upper = sps.vstack(( rate_batt * one_n1,
                                rate_h2 * one_n1,
                                max_soc*one_np1_1,
                                max_h2*one_np1_1,
                                rate_batt*one_11,
                                max_soc*one_11,
                                rate_h2*one_11,
                                max_h2*one_11)).toarray().squeeze()


    return mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper

def solve_lp_sparse_sf(price_ts: TimeSeries, prod_wind: Production,
                    prod_pv: Production, stor_batt: Storage, stor_h2: Storage,
                    discount_rate: float, n_year: int,
                    p_min: float | np.ndarray, p_max: float,
                    n: int) -> OpSchedule:
    """Build and solve a LP for NPV maximization.

    This function builds and solves the hybrid sizing and operation
    problem as a linear program, in a short formulation (sf). The
    objective is to minimize the Net Present Value of the plant. In this
    function, the input for the power production represented by two
    Production objects, one for wind and one for solar.

    Params:
        price_ts (TimeSeries): Time series of the price of electricity
            on theday-ahead market [currency/MWh].
        prod_wind (Production): Object representing the power production
            from wind energy system.
        prod_pv (Production): Object representing the power production
            from solar PV system.
        stor_batt (Storage): Object describing the battery storage.
        stor_h2 (Storage): Object describing the hydrogen storage system.
        discount_rate (float): Discount rate for the NPV calculation [-].
        n_year (int): Number of years for the NPV calculation [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational
            schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match, if the length of the power in the
            Production objects is below n.
        RuntimeError: if the optimization algorithm fails to solve the
            problem.
    """

    dt = prod_wind.power.dt

    assert dt == price_ts.dt
    assert dt == prod_pv.power.dt
    assert n <=  len(prod_wind.power.data)
    assert n <=  len(prod_pv.power.data)
    assert n <=  len(price_ts.data)

    power_res = prod_wind.power.data[:n] + prod_pv.power.data[:n]

    eps_batt = 1 - stor_batt.eff_in * stor_batt.eff_out
    eps_h2 = 1 - stor_h2.eff_in * stor_h2.eff_out

    vec_obj = build_lp_obj_npv_sf(price_ts.data, n, stor_batt.p_cost,
                               stor_batt.e_cost, stor_h2.p_cost,
                               stor_h2.e_cost, discount_rate, n_year)


    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper = \
        build_lp_cst_sparse_sf(power_res, dt, p_min, p_max, n, eps_batt, eps_h2,
                            rate_batt = stor_batt.p_cap, rate_h2 = stor_h2.p_cap,
                            max_soc = stor_batt.e_cap, max_h2= stor_h2.e_cap)

    n_var = bounds_upper.shape[0]
    n_cstr_eq = vec_eq.shape[0]
    n_cstr_ineq = vec_ineq.shape[0]

    assert n_var == bounds_lower.shape[0]
    assert n_var == vec_obj.shape[0]

    bounds = []
    for x in range(0, n_var):
        bounds.append((bounds_lower[x], bounds_upper[x]))

    try:
        # x = linprog_mosek(n_var, n_cstr_eq, n_cstr_ineq, mat_eq, vec_eq, mat_ineq,
        #  vec_ineq, vec_obj, bounds_lower, bounds_upper)
        res = linprog(vec_obj, A_ub= mat_ineq.toarray(), b_ub = vec_ineq, A_eq=mat_eq.toarray(),
                  b_eq=vec_eq, bounds=bounds, method = 'highs')
        x = res.x
    except:
        traceback.print_exc()
        raise RuntimeError from None

    if res.status != 0:
        print(res.message)
        raise RuntimeError
    
    power_batt = x[0:n]
    power_h2 = x[n:2*n]
    soc = x[2*n:3*n+1]
    h2 = x[3*n+1:4*n+2]
    batt_p_cap = x[4*n+2]
    batt_e_cap = x[4*n+3]
    h2_p_cap = x[4*n+4]
    h2_e_cap = x[4*n+5]

    power_res_new = []
    power_losses_bat = []
    power_losses_h2 = []
    for i in range(n):
        power_res_new.append(min(p_max - power_batt[i] - power_h2[i],
                             power_res[i]))

        power_losses_bat.append(-(soc[i+1] - soc[i] + dt*power_batt[i])/dt)
        power_losses_h2.append(-(h2[i+1] - h2[i] + dt*power_h2[i])/dt)

    stor_batt_res = Storage(e_cap = batt_e_cap,
                            p_cap = batt_p_cap,
                            eff_in = 1,
                            eff_out = 1-eps_batt,
                            p_cost = stor_batt.p_cost,
                            e_cost = stor_batt.e_cost)
    stor_h2_res = Storage(e_cap = h2_e_cap,
                            p_cap = h2_p_cap,
                            eff_in = 1,
                            eff_out = 1-eps_h2,
                            p_cost = stor_h2.p_cost,
                            e_cost = stor_h2.e_cost)

    prod_wind_res = Production(power_ts = TimeSeries(np.array(power_res_new) - prod_pv.power.data[:n], dt), p_cost= prod_wind.p_cost)

    os_res = OpSchedule(production_list = [prod_wind_res, prod_pv],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(prod_wind_res.power.data[:n], dt),
                                        TimeSeries(prod_pv.power.data[:n], dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc[:n], dt),
                                     TimeSeries(h2[:n], dt)],
                        price = price_ts.data[:n])

    os_res.get_npv_irr(discount_rate, n_year)

    os_res.losses = [np.array(power_losses_bat) ,  np.array(power_losses_h2)]

    return os_res

def os_rule_based(price_ts: TimeSeries, prod_wind: Production,
                  prod_pv: Production, stor_batt: Storage, stor_h2: Storage,
                  discount_rate: float, n_year: int, p_min: float | np.ndarray,
                  p_rule: float, price_min: float,
                  n: int, e_start: float = 0) -> OpSchedule:

    """Build the operation schedule following a rule-based control.

    This function builds the operation schedule for a hybrid power plant
    following a rule-based approach. The objective of the controller is
    to satisfy a baseload power represented by p_min.
    The control rules are as follow:
        - if the power from wind and pv is above a given value (p_rule),
        the storage systems are charged: first the battery (short-term)
        and then the hydrogen system (long-term).
        - if the power from wind and pv is below p_rule but above the
        baseload, and if the price is above a threshold (price_min), the
        storage systems should sell power
        - if the power output is below the required baseload, power is
        delivered from the storage systems: first long-term, then the
        short-term one.


    This implementation is based on the work by Jasper Kreeft for the
    sizing of the Baseload Power Hub.

    Params:
        price_ts (TimeSeries): Time series of the price of electricity
            on theday-ahead market [currency/MWh].
        prod_wind (Production): Object describing the wind production.
        prod_pv (Production): Object describing the solar pv production.
        stor_batt (Storage): Object describing the battery storage.
        stor_h2 (Storage): Object describing the hydrogen storage system.
        discount_rate (float): Discount rate for the NPV calculation [-].
        n_year (int): Number of years for the NPV calculation [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_rule (float): Power above which the storage should charge [MW].
        price_min (float): Price above which the storage should
            discharge [currency]
        n (int): Number of time steps to consider in the optimization.

    Returns:
        os_res (OpSchedule): Object describing the operational schedule.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match.
        RuntimeError: if the optimization algorithm fails to solve the
            problem.

    """



    dt = prod_wind.power.dt
    assert prod_pv.power.dt == dt

    power_res = prod_wind.power.data[:n] + prod_pv.power.data[:n]

    soc_batt = np.zeros((n+1,))
    soc_batt[0] = e_start
    soc_h2 = np.zeros((n+1,))
    power_batt = np.zeros((n,))
    power_h2 = np.zeros((n,))

    p_max = max(power_res)
    rate_h2_min = 0.0*p_max
    # p_sb = 0.0  #standby power, unused
    p_mid = 10*p_max  #electrolyzer efficiency reduced abpve p_mid
    tmp_slope = 1.0 #0.8
    tmp_cst = -(tmp_slope-1) * p_mid * dt

    for t in range(0,n):

        avail_power = power_res[t] - p_rule

        if avail_power>=0:

            power_batt[t] = max(-stor_batt.p_cap,
                            -(stor_batt.e_cap-soc_batt[t])/dt/stor_batt.eff_in,
                            -avail_power )

            avail_power += power_batt[t]  # power_res is <0

            power_h2[t] = max(-stor_h2.p_cap,
                              -(stor_h2.e_cap - soc_h2[t])/dt/stor_h2.eff_in,
                              -avail_power )

            avail_power += power_h2[t]



        elif power_res[t]  >= p_min:
            power_batt[t] = 0
            power_h2[t] = 0
            #if the price is high enough, sell as much as posible
            if price_ts.data[t]>price_min:
                if soc_h2[t]>0:
                    power_h2[t] = min(stor_h2.p_cap,
                                      soc_h2[t]/dt * stor_h2.eff_out)
                if soc_batt[t]>0:
                    power_batt[t] = min(stor_batt.p_cap,
                                        soc_batt[t]/dt*stor_batt.eff_out)

        else:
            missing_power = p_min - power_res[t ]

            if soc_h2[t]>0:
                power_h2[t] = min(stor_h2.p_cap,
                                  soc_h2[t]/dt*stor_h2.eff_out, missing_power)
            else:
                power_h2[t] = 0

            missing_power -= power_h2[t]


            if soc_batt[t]>0:
                power_batt[t] = min(stor_batt.p_cap,
                                    soc_batt[t]/dt*stor_batt.eff_out,
                                    missing_power)
            else:
                power_batt[t] = 0




        if power_batt[t] >= 0:
            soc_batt[t+1] = soc_batt[t] \
                            - dt*(power_batt[t])/stor_batt.eff_out
        else:
            soc_batt[t+1] = soc_batt[t] \
                            - dt*(power_batt[t])*stor_batt.eff_in

        if power_h2[t] <= - p_mid / stor_h2.eff_out:
            ## lower efficiency ## power_res <0 and losses>0
            soc_h2[t+1] = soc_h2[t] + tmp_cst \
                        - tmp_slope * dt * (power_h2[t]) * stor_h2.eff_in
        elif power_h2[t] <= -rate_h2_min:
            # soc_h2[t+1] = soc_h2[t] - dt * (power_h2[t] - losses_h2[t])
            ## power_res <0 and losses>0
            soc_h2[t+1] = soc_h2[t] - dt  *(power_h2[t]) * stor_h2.eff_in
        elif power_h2[t] >= 0:
            # soc_h2[t+1] = soc_h2[t] - dt * (power_h2[t] - losses_h2[t])
            ## power_res>0 ands losses >0
            soc_h2[t+1] = soc_h2[t] - dt * (power_h2[t]) / stor_h2.eff_out
        else:
            soc_h2[t+1] = soc_h2[t]

    stor_batt_res = Storage(e_cap = max(soc_batt),
                            p_cap = max(power_batt),
                            eff_in = stor_batt.eff_in,
                            eff_out = stor_batt.eff_out,
                            p_cost = stor_batt.p_cost,
                            e_cost = stor_batt.e_cost)

    #find minimum storage from the maximum discharge cycle
    soc_h2_max = max(soc_h2)
    # import rainflow
    # rng_vec = []
    # for rng, mn, count, i_start, i_end in rainflow.extract_cycles(soc_h2):
    #     if soc_h2[i_start] - soc_h2[i_end] > 0:
    #         rng_vec.append(rng)
    # if len(rng_vec)>0:
    #     soc_h2_max = max(rng_vec)

    stor_h2_res = Storage(e_cap = soc_h2_max,
                            p_cap = max(power_h2),
                            eff_in = stor_h2.eff_in,
                            eff_out = stor_h2.eff_out,
                            p_cost = stor_h2.p_cost,
                            e_cost = stor_h2.e_cost)


    os_res = OpSchedule(production_list = [prod_wind, prod_pv],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [prod_wind.power, prod_pv.power],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc_batt, dt),
                                     TimeSeries(soc_h2, dt)],
                        price = price_ts.data[:n])

    os_res.get_npv_irr(discount_rate, n_year)

    return os_res
