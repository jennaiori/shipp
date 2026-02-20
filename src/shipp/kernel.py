"""This module defines kernel functions for shipp.

The functions defined in this module are used to solve the dispatch optimization problem with scipy.optimize.linprog or with a rule-based approach. They build the matrices describing the linear optimization from scratch. For complex problems, the use of the functions in kernel_pyomo is prefered.

Functions:
    - build_lp_obj_npv: Build objective vector for NPV maximization.
    - build_lp_cst_sparse: Build sparse constraints for a LP.
    - solve_lp_sparse: Build and solve a LP for NPV maximization.
    - os_rule_based: Build the operation schedule with a rule-based EMS
"""

import traceback
import numpy as np
import numpy_financial as npf
from scipy.optimize import linprog
import scipy.sparse as sps

from shipp.components import Storage, OpSchedule, Production
from shipp.timeseries import TimeSeries

TOL = 1e-4 # tolerance for checking the losses of the storage system

def build_lp_obj_npv(price: np.ndarray, n: int, stor1_p_cost: float, stor1_e_cost: float, stor2_p_cost: float, stor2_e_cost: float,
                     discount_rate: float, n_year: int) -> np.ndarray:
    """Build the objective function vector for NPV maximization for the LP formulation.

    This function returns an objective vector corresponding the
    maximization of Net Present Value (NPV):

        f(x) = - factor*price*power + stor1_e_cost*stor1_e_cap + stor1_p_cost*stor1_p_cap + stor2_e_cost*stor2_e_cap + stor2_p_cost*stor2_p_cap

    where factor = sum_n=1^(n_year) (1+discount_rate)**(-n).
    The objective function vector corresponds to the following design variables:
        - Power from storage 1, shape-(n,)
        - Power from storage 2, shape-(n,)
        - State of charge (energy) of storage 1, shape-(n+1,)
        - State of charge (energy) of storage 2, shape-(n+1,)
        - Power capacity of storage 1, shape-(1,)
        - Energy capacity of storage 1, shape-(1,)
        - Power capacity of storage 2, shape-(1,)
        - Energy capacity of storage 2, shape-(1,)

    The number of design variables is n_x = 4*n+6

    Args:
        price (np.array): An array of electricity price to calculate the revenue [currency/MWh]
        n (int): the number of time steps [-]
        stor1_p_cost (float): cost of storage 1 per power capacity [currency/MW]
        stor1_e_cost (float): cost of storage 1 per energy capacity [currency/MWh]
        stor2_p_cost (float): cost of storage 2 per power capacity [currency/MW]
        stor2_e_cost (float): cost of storage 2 per energy capacity [currency/MWh]
        discount_rate (float): discount rate to calculate the NPV [-]
        n_year (int): number of years of operation of the project [-]

    Returns:
        np.ndarray: A shape-(n_x,) array representing the objective function of the linear program [-]

    Raises:
        AssertionError: if the length of the price is below n, if any input is not finite
    """

    assert len(price) >= n
    assert np.all(np.isfinite(price))
    assert n != 0
    assert np.isfinite(stor1_p_cost)
    assert np.isfinite(stor1_e_cost)
    assert np.isfinite(stor2_p_cost)
    assert np.isfinite(stor2_e_cost)
    assert n_year > 0
    assert 0 <= discount_rate <= 1


    factor = npf.npv(discount_rate, np.ones(n_year))-1

    normed_price = 365 * 24 / n * np.reshape(price[:n], (n,1))*factor

    vec_obj = np.vstack((-normed_price,             # Batt power
                        -normed_price,             # Power from fuel cell / to electrolyzer
                        np.zeros((n+1, 1)),
                        np.zeros((n+1, 1)),
                        stor1_p_cost*np.ones((1,1)),           # minimize max batt power
                        stor1_e_cost*np.ones((1,1)),           # minimize max state of charge
                        stor2_p_cost*np.ones((1,1)),             # minimize max stor2_e power rate
                        stor2_e_cost*np.ones((1,1)))).squeeze()  # minimize max stor2_e energy capacity

    return vec_obj

def build_lp_cst_sparse(power: np.ndarray, dt: float, p_min, p_max: float, n: int, stor1_eff: float, stor2_eff: float, stor1_p_cap_max: float = -1.0, stor2_p_cap_max: float = -1.0, stor1_e_cap_max: float = -1.0, stor2_e_cap_max: float = -1.0, fixed_cap = False ) -> tuple[sps.coo_matrix, np.ndarray, sps.coo_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Build the sparse constraints for the LP formulation of the dispatch optimization problem.

    Function to build the matrices and vectors for the constraints of the dispatch optimization problem, considering two different storage systems, and as a linear program. A sparse format is used to represent the matrices.

    The constraints are made of equality and inequality  constraints such that:

        - mat_eq * x = vec_eq
        - mat_ineq * x <= vec_ineq
        - bounds_lower <= x <= bounds_upper
  
    With n the number of time steps, the problem is made of:

        - n_x = 4*n+6 design variables
        - n_eq = 2 equality constraints
        - n_ineq = 12*n+2 inequality constraints

    The design variables for the linear problem are:

        - Power from storage 1, shape-(n,)
        - Power from storage 2, shape-(n,)
        - State of charge (energy) of storage 1, shape-(n+1,)
        - State of charge (energy) of storage 2, shape-(n+1,)
        - Power capacity of storage 1, shape-(1,)
        - Energy capacity of storage 1, shape-(1,)
        - Power capacity of storage 2, shape-(1,)
        - Energy capacity of storage 2, shape-(1,)

    The equality constraints for the problem are:

        - Constraint to enforce the value of the first stored energy of storage 1 is equal to the last (size 1)
        - Constraint to enforce the value of the first stored energy of storage 2 is equal to the last (size 1)

    The inequality constraints for the problem are:

        - Constraints on the minimum and maximum combined power from production and storage assets (size 2*n)
        - Constraints on the stored energy of storage 1 (size 2*n) 
        - Constraints on the stored energy of storage 2 (size 2*n)
        - Constraints on the maximmum and minimum power to and from storage 1 (size 2*n)
        - Constraints on the maximmum and minimum power to and from storage 2 (size 2*n)
        - Constraints on the maximum stored energy in storage 1 (size n+1)
        - Constraints on the maximum stored energy in storage 2 (size n+1)

    Args:
        power (np.ndarray): A shape-(n,) array for the power production from renewables [MW].
        dt (float): time step [s].
        p_min (float or np.ndarray): A float or shape-(n,) array for the minimum required power production [MW].
        p_max (float): maximum power production [MW]
        n (int): number of time steps [-].
        stor1_eff (float): represents the round trip efficiency of storage 1 [-]. The losses are applied in discharge.
        stor2_eff (float): represents the round trip efficiency of storage 2 [-]. The losses are applied in discharge..
        stor1_p_cap_max (float): maximum power capacity for storage 1 [MW]. Default to -1.0 when there is no limit in power rate.
        stor2_p_cap_max (float): maximum power capacity for storage 2 [MW]. Default to -1.0 when there is no limit in power rate.
        stor1_e_cap_max (float): maximum energy capacity for storage 1 [MWh]. Default to -1.0 when there is no limit.
        stor2_e_cap_max (float): maximum energy capacity for storage 2 [MWh]. Default to -1.0 when there is no limit.

    Returns:
        tuple[sps.coo_matrix, np.ndarray, sps.coo_matrix, np.ndarray, np.ndarray, np.ndarray] : [mat_eq, vec_ec, mat_ineq, vec_ineq, bounds_lower, bounds_upper] matrices and vectors representing the inequality, equality and bound constraints of the problem.

    Raises:
        ValueError: if argument p_min is not a float or a list of floats
        AssertionError: if any argument is not finite and if the argument power has a length lower than n
    """

    assert np.all(np.isfinite(power))
    assert len(power) >= n
    assert np.all(np.isfinite(p_min))
    assert np.isfinite(p_max)
    assert np.isfinite(dt)
    assert np.isfinite(stor1_eff)
    assert np.isfinite(stor2_eff)

    if stor1_p_cap_max == -1 or stor1_p_cap_max is None:
        stor1_p_cap_max = p_max

    if stor2_p_cap_max == -1 or stor2_p_cap_max is None:
        stor2_p_cap_max = p_max

    if stor1_e_cap_max == -1 or stor1_e_cap_max is None:
        stor1_e_cap_max = n*dt*stor1_p_cap_max #MWh

    if stor2_e_cap_max == -1 or stor2_e_cap_max is None:
        stor2_e_cap_max = n*dt*stor2_p_cap_max #MWh  (1400kg H2 and 1kg is 33.3 kWh)

    assert np.isfinite(stor1_p_cap_max)
    assert np.isfinite(stor2_p_cap_max)
    assert np.isfinite(stor1_e_cap_max)
    assert np.isfinite(stor2_e_cap_max)

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
    # Constraint on first stored energy of storage 1
    mat_stor1_first_e = sps.hstack((z_1n, z_1n,
                            one_11, np.zeros((1, n-1)), -one_11,
                            z_1_np1,
                            z_11, z_11, z_11, z_11))
    vec_stor1_first_e = z_11

    # Constraint on first stored energy of storage 2
    mat_stor2_first_e = sps.hstack((z_1n, z_1n,
                            z_1_np1,
                            one_11, np.zeros((1, n-1)), -one_11,
                            z_11, z_11, z_11, z_11))
    vec_stor2_first_e = z_11

    # INEQ CONSTRAINT
    # Constraint on power production + storage 1 power + storage 2 power >= p_min
    mat_power_bound = sps.hstack((eye_n, eye_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    vec_power_min = p_min_vec - power[:n].reshape(n,1)
    vec_power_max = power_ub

    # Constraint on the maximum stored energy of storage 1
    mat_stor1_max_energy = sps.hstack((z_np1_n, z_np1_n, eye_np1, z_np1,
                                z_np1_1, -1*one_np1_1, z_np1_1, z_np1_1))
    vec_stor1_max_energy = z_np1_1

    # Constraints on the minimum and maximum power of storage 1
    mat_stor1_max_power = sps.hstack((eye_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
    vec_stor1_max_power = z_n1

    mat_stor1_min_power = sps.hstack(( -eye_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
    vec_stor1_min_power = z_n1

    # Constraint on the maximum stored energy of storage 2
    mat_stor2_max_energy = sps.hstack((z_np1_n, z_np1_n, z_np1,  eye_np1,
                             z_np1_1, z_np1_1, z_np1_1, -1*one_np1_1,))
    vec_stor2_max_energy = z_np1_1

    # Constraint on minimum and maximum power of storage 2
    mat_stor2_max_power = sps.hstack((z_n, eye_n, z_n_np1, z_n_np1,
                                   z_n1, z_n1, -one_n1, z_n1))
    vec_stor2_max_power = z_n1

    mat_stor2_min_power = sps.hstack((z_n, -eye_n, z_n_np1, z_n_np1,
                                   z_n1, z_n1, -one_n1, z_n1))
    vec_stor2_min_power = z_n1

    # Constraint representing the storage model, linking stored energy to power and including storage losses
    # e_(n+1) - e_(n) <= - dt * p_(n)
    mat_stor1_model_in = sps.hstack((dt * eye_n, z_n,
                            -mat_diag_soc, -mat_last_soc, z_n_np1,
                            z_n1, z_n1, z_n1, z_n1))
    vec_stor1_model_in = z_n1

    # e_(n+1) - e_(n) <= - dt/eta * p_(n)
    mat_stor1_model_out = sps.hstack((dt/stor1_eff * eye_n, z_n,
                            -mat_diag_soc, -mat_last_soc, z_n_np1,
                            z_n1, z_n1, z_n1, z_n1))
    vec_stor1_model_out = z_n1

    # Constraint on hydrogen levels:
    #  e_(n+1) - e_(n) <= - dt * p_(n)
    mat_stor2_model_in = sps.hstack((z_n, dt * eye_n,
                                z_n_np1, -mat_diag_soc, -mat_last_soc,
                                z_n1, z_n1, z_n1, z_n1))
    vec_stor2_model_in = z_n1

    #  e_(n+1) - e_(n) <= - dt/eta * p_(n)
    mat_stor2_model_out = sps.hstack((z_n, dt/stor2_eff * eye_n,
                                z_n_np1, -mat_diag_soc, -mat_last_soc,
                                z_n1, z_n1, z_n1, z_n1))
    vec_stor2_model_out = z_n1

    ## Assemble matrices
    mat_eq = sps.vstack((mat_stor1_first_e,  mat_stor2_first_e))
    vec_eq = sps.vstack((vec_stor1_first_e,  vec_stor2_first_e)).toarray().squeeze()

    mat_ineq = sps.vstack((-1*mat_power_bound, mat_power_bound,
                                mat_stor1_model_in, mat_stor1_model_out,
                                mat_stor2_model_in, mat_stor2_model_out,
                                mat_stor1_max_energy, mat_stor2_max_power,
                                mat_stor2_min_power, mat_stor1_max_power,
                                mat_stor1_min_power, mat_stor2_max_energy))
    vec_ineq = sps.vstack((-1*vec_power_min, vec_power_max,
                                vec_stor1_model_in, vec_stor1_model_out,
                                vec_stor2_model_in, vec_stor2_model_out,
                                vec_stor1_max_energy, vec_stor2_max_power,
                                vec_stor2_min_power, vec_stor1_max_power,
                                vec_stor1_min_power, vec_stor2_max_energy)).toarray().squeeze()
    # BOUNDS ON DESIGN VARIABLES
    if fixed_cap == False:
        bounds_lower = sps.vstack((-stor1_p_cap_max * one_n1,
                                    -stor2_p_cap_max * one_n1,
                                    z_np1_1,
                                    z_np1_1,
                                    z_11,
                                    z_11,
                                    z_11,
                                    z_11)).toarray().squeeze()
    else:
        bounds_lower = sps.vstack((-stor1_p_cap_max * one_n1,
                                    -stor2_p_cap_max * one_n1,
                                    z_np1_1,
                                    z_np1_1,
                                    stor1_p_cap_max*one_11,
                                    stor1_e_cap_max*one_11,
                                    stor2_p_cap_max*one_11,
                                    stor2_e_cap_max*one_11)).toarray().squeeze()

    bounds_upper = sps.vstack(( stor1_p_cap_max * one_n1,
                                stor2_p_cap_max * one_n1,
                                stor1_e_cap_max*one_np1_1,
                                stor2_e_cap_max*one_np1_1,
                                stor1_p_cap_max*one_11,
                                stor1_e_cap_max*one_11,
                                stor2_p_cap_max*one_11,
                                stor2_e_cap_max*one_11)).toarray().squeeze()


    return mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper

def solve_lp_sparse(price_ts: TimeSeries, prod1: Production,
                    prod2: Production, stor1: Storage, stor2: Storage,
                    discount_rate: float, n_year: int,
                    p_min, p_max: float,
                    n: int, fixed_cap: bool = False) -> OpSchedule:
    """Build and solve the integrated dispatch optimization problem, formulated as a linear program.

    This function builds and solves the hybrid sizing and operation problem as a linear program. The objective is to minimize the Net Present Value of the plant. The optimization problem finds the optimal energy and power capacity of two storage systems and their optimal dispatch. In this function, the power production inputs are represented by two Production objects (e.g. one for wind and one for solar PV).

    Args:
        price_ts (TimeSeries): Time series of the price of electricity on the day-ahead market [currency/MWh].
        prod1 (Production): Object representing the power production of the first production asset.
        prod2 (Production): Object representing the power production of the second production asset.
        stor1 (Storage): Object describing storage 1.
        stor2 (Storage): Object describing storage 2.
        discount_rate (float): Discount rate for the NPV calculation [-].
        n_year (int): Number of years for the NPV calculation [-].
        p_min (float or np.ndarray): Minimum power requirement (e.g. baseload) [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.
        fixed_cap (bool): If True, the capacity of the storage is fixed.

    Returns:
        OpSchedule: Object describing the optimal operational schedule and optimal storage capacities.

    Raises:
        AssertionError: if the time step of the power and price time series do not match, if the length of the power in the Production objects is below n.
        RuntimeError: if the optimization algorithm fails to solve the problem.
    """

    dt = prod1.power.dt

    assert dt == price_ts.dt
    assert dt == prod2.power.dt
    assert n <=  len(prod1.power.data)
    assert n <=  len(prod2.power.data)
    assert n <=  len(price_ts.data)

    assert (stor1.dod == 1) and (stor2.dod == 1), "solve_lp_sparse is not implemented for storage depth of charge below 100%"

    power_res = prod1.power.data[:n] + prod2.power.data[:n]

    stor1_eff = stor1.eff_in * stor1.eff_out
    stor2_eff = stor2.eff_in * stor2.eff_out

    # Build the vector representing the objective function 
    vec_obj = build_lp_obj_npv(price_ts.data, n, stor1.p_cost, stor1.e_cost, stor2.p_cost, stor2.e_cost, discount_rate, n_year)

    # Build the matrices and vectors representing the constraints of the problem  
    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper =  build_lp_cst_sparse(power_res, dt, p_min, p_max, n, stor1_eff, stor2_eff, stor1_p_cap_max = stor1.p_cap, stor2_p_cap_max = stor2.p_cap, stor1_e_cap_max = stor1.e_cap, stor2_e_cap_max= stor2.e_cap, fixed_cap = fixed_cap)

    n_var = bounds_upper.shape[0]
    n_cstr_eq = vec_eq.shape[0]
    n_cstr_ineq = vec_ineq.shape[0]

    assert n_var == bounds_lower.shape[0]
    assert n_var == vec_obj.shape[0]

    bounds = []
    for x in range(0, n_var):
        bounds.append((bounds_lower[x], bounds_upper[x]))

    # Solve the problem using linprog
    try:
        res = linprog(vec_obj, A_ub= mat_ineq.toarray(), b_ub = vec_ineq, A_eq=mat_eq.toarray(), b_eq=vec_eq, bounds=bounds, method = 'highs')
        x = res.x
    except:
        traceback.print_exc()
        raise RuntimeError from None

    if res.status != 0:
        print(res.message)
        raise RuntimeError
    
    # Extract solution
    stor1_p = x[0:n]
    stor2_p = x[n:2*n]
    stor1_e = x[2*n:3*n+1]
    stor2_e = x[3*n+1:4*n+2]
    stor1_p_cap = x[4*n+2]
    stor1_e_cap = x[4*n+3]
    stor2_p_cap = x[4*n+4]
    stor2_e_cap = x[4*n+5]

    power_res_new = []
    power_losses_stor1 = []
    power_losses_stor2 = []


    for i in range(n):
        # Compute the power produced by the first production asset, and remove curtailment
        power_res_new.append(min(p_max - stor1_p[i] - stor2_p[i], power_res[i]))
        
        # Calculate the losses in the solution
        power_losses_stor1.append(-(stor1_e[i+1] - stor1_e[i] + dt*stor1_p[i])/dt)
        power_losses_stor2.append(-(stor2_e[i+1] - stor2_e[i] + dt*stor2_p[i])/dt)

    stor1_res = Storage(e_cap = stor1_e_cap,
                            p_cap = stor1_p_cap,
                            eff_in = 1,
                            eff_out = stor1_eff,
                            p_cost = stor1.p_cost,
                            e_cost = stor1.e_cost,
                            dod = stor1.dod)
    stor2_res = Storage(e_cap = stor2_e_cap,
                            p_cap = stor2_p_cap,
                            eff_in = 1,
                            eff_out = stor2_eff,
                            p_cost = stor2.p_cost,
                            e_cost = stor2.e_cost,
                            dod = stor2.dod)

    prod1_res = Production(power_ts = TimeSeries(np.array(power_res_new) - prod2.power.data[:n], dt), p_cost= prod1.p_cost)

    os_res = OpSchedule(production_list = [prod1_res, prod2],
                        storage_list = [stor1_res, stor2_res],
                        production_p = [TimeSeries(prod1_res.power.data[:n], dt),
                                        TimeSeries(prod2.power.data[:n], dt)],
                        storage_p = [TimeSeries(stor1_p, dt),
                                     TimeSeries(stor2_p, dt)],
                        storage_e = [TimeSeries(stor1_e[:n], dt),
                                     TimeSeries(stor2_e[:n], dt)],
                        price = price_ts.data[:n])

    os_res.get_npv_irr(discount_rate, n_year)

    os_res.losses = [np.array(power_losses_stor1) ,  np.array(power_losses_stor2)]

    return os_res

def os_rule_based(price_ts: TimeSeries, prod1: Production, prod2: Production, stor1: Storage, stor2: Storage, discount_rate: float, n_year: int, p_min, p_rule: float, price_min: float, n: int, e_start: float = 0) -> OpSchedule:

    """Build the operation schedule following a rule-based control.

    This function builds the operation schedule for a hybrid power plant following a rule-based approach. The objective of the controller is to satisfy a baseload power represented by p_min. The control rules are as follow:

        - if the power produced is above a given value (p_rule), the storage systems are charged.
        - if the power produced is below p_rule but above the baseload, and if the price is above a threshold (price_min), the storage systems should sell power
        - if the power output is below the required baseload, power is delivered from the storage systems.

    This implementation is based on the work by Jasper Kreeft for the sizing of the Baseload Power Hub.

    Args:
        price_ts (TimeSeries): Time series of the price of electricity on theday-ahead market [currency/MWh].
        prod1 (Production): Object describing the first production asset.
        prod2 (Production): Object describing the second production asset.
        stor1 (Storage): Object describing storage 1.
        stor2 (Storage): Object describing storage 2.
        discount_rate (float): Discount rate for the NPV calculation [-].
        n_year (int): Number of years for the NPV calculation [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_rule (float): Power above which the storage should charge [MW].
        price_min (float): Price above which the storage should discharge [currency]
        n (int): Number of time steps to consider simulation.

    Returns:
        OpSchedule: Object describing the operational schedule.

    Raises:
        AssertionError: if the time step of the power and price time series do not match.
    """

    dt = prod1.power.dt
    assert prod2.power.dt == dt

    power_res = prod1.power.data[:n] + prod2.power.data[:n]

    stor1_e = np.zeros((n+1,))
    stor1_e[0] = e_start
    stor2_e = np.zeros((n+1,))
    stor1_p = np.zeros((n,))
    stor2_p = np.zeros((n,))

    p_max = max(power_res)

    stor2_rate_min = 0.0*p_max
    p_mid = 10*p_max  #storage 2 efficiency reduced above p_mid
    tmp_slope = 1.0 #0.8
    tmp_cst = -(tmp_slope-1) * p_mid * dt

    for t in range(0,n):

        avail_power = power_res[t] - p_rule

        if avail_power>=0:  
            # Charge storage 1 first
            stor1_p[t] = max(-stor1.p_cap,
                            -(stor1.e_cap-stor1_e[t])/dt/stor1.eff_in,
                            -avail_power )

            avail_power += stor1_p[t]  # this operation "reduces" the available power since stor1_p is <0

            # Charge storage 2 next
            stor2_p[t] = max(-stor2.p_cap,
                              -(stor2.e_cap - stor2_e[t])/dt/stor2.eff_in,
                              -avail_power )

            avail_power += stor2_p[t]

        elif power_res[t]  >= p_min:
            stor1_p[t] = 0
            stor2_p[t] = 0
            #if the price is high enough, discharge storage 1 to sell as much as posible
            if price_ts.data[t] > price_min:
                if stor2_e[t] > stor2.e_cap*(1 - stor2.dod):
                    stor2_p[t] = min(stor2.p_cap,
                                      (stor2_e[t] - stor2.e_cap*(1 - stor2.dod))/dt * stor2.eff_out)
                if stor1_e[t] > stor1.e_cap*(1 - stor1.dod):
                    stor1_p[t] = min(stor1.p_cap,
                                        (stor1_e[t] - stor1.e_cap*(1 - stor1.dod))/dt*stor1.eff_out)

        else:
            # If the power produced is below the required baseload, discharge the storage systems
            missing_power = p_min - power_res[t ]

            if stor2_e[t]>  stor2.e_cap*(1 - stor2.dod):
                stor2_p[t] = min(stor2.p_cap,
                                  (stor2_e[t] -  stor2.e_cap*(1 - stor2.dod))/dt*stor2.eff_out, missing_power)
            else:
                stor2_p[t] = 0

            missing_power -= stor2_p[t]


            if stor1_e[t]> stor1.e_cap*(1 - stor1.dod):
                stor1_p[t] = min(stor1.p_cap,
                                    (stor1_e[t] - stor1.e_cap*(1 - stor1.dod))/dt*stor1.eff_out,
                                    missing_power)
            else:
                stor1_p[t] = 0

        # Calculate the state of the storage systems for the next time steps
        if stor1_p[t] >= 0:
            stor1_e[t+1] = stor1_e[t] \
                            - dt*(stor1_p[t])/stor1.eff_out
        else:
            stor1_e[t+1] = stor1_e[t] \
                            - dt*(stor1_p[t])*stor1.eff_in

        if stor2_p[t] <= - p_mid / stor2.eff_out:
            ## lower efficiency ## power_res <0 and losses>0
            stor2_e[t+1] = stor2_e[t] + tmp_cst \
                        - tmp_slope * dt * (stor2_p[t]) * stor2.eff_in
        elif stor2_p[t] <= -stor2_rate_min:
            stor2_e[t+1] = stor2_e[t] - dt  *(stor2_p[t]) * stor2.eff_in
        elif stor2_p[t] >= 0:
            stor2_e[t+1] = stor2_e[t] - dt * (stor2_p[t]) / stor2.eff_out
        else:
            stor2_e[t+1] = stor2_e[t]

    stor1_res = Storage(e_cap = max(stor1_e),
                            p_cap = max(stor1_p),
                            eff_in = stor1.eff_in,
                            eff_out = stor1.eff_out,
                            p_cost = stor1.p_cost,
                            e_cost = stor1.e_cost,
                            dod = stor1.dod)

    stor2_res = Storage(e_cap = max(stor2_e),
                            p_cap = max(stor2_p),
                            eff_in = stor2.eff_in,
                            eff_out = stor2.eff_out,
                            p_cost = stor2.p_cost,
                            e_cost = stor2.e_cost,
                            dod = stor2.dod)

    os_res = OpSchedule(production_list = [prod1, prod2],
                        storage_list = [stor1_res, stor2_res],
                        production_p = [prod1.power, prod2.power],
                        storage_p = [TimeSeries(stor1_p, dt),
                                     TimeSeries(stor2_p, dt)],
                        storage_e = [TimeSeries(stor1_e, dt),
                                     TimeSeries(stor2_e, dt)],
                        price = price_ts.data[:n])

    os_res.get_npv_irr(discount_rate, n_year)

    return os_res
