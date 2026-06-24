"""This module defines kernel functions for shipp.

The functions defined in this module are used to solve the dispatch optimization problem with scipy.optimize.linprog or with a rule-based approach. They build the matrices describing the linear optimization from scratch. For complex problems, the use of the functions in kernel_pyomo is prefered.

Functions:
    - build_lp_obj_npv: Build objective vector for NPV maximization.
    - build_lp_cst_sparse: Build sparse constraints for a LP.
    - solve_lp_sparse: Build and solve a LP for NPV maximization.
    - os_rule_based: Build the operation schedule with a rule-based EMS
    - financial_metrics: Calculate financial metrics for a given hybrid power plant
"""

import traceback
import numpy as np
import numpy_financial as npf
from scipy.optimize import linprog
import scipy.sparse as sps

from shipp.components import Storage, OpSchedule, Production
from shipp.timeseries import TimeSeries

TOL = 1e-4 # tolerance for checking the losses of the storage system
DEFAULT_ALPHA_OBJ = (1-1e-6)
BIG_M = 100

def build_lp_obj_npv(price: np.ndarray, n: int, stor1_p_cost: float, stor1_e_cost: float, stor2_p_cost: float, stor2_e_cost: float, discount_rate: float, n_year: int, options: dict = None) -> np.ndarray:
    """Build the objective function vector for NPV maximization for the LP formulation.

    This function returns an objective vector corresponding to the
    maximization of Net Present Value (NPV):

        f(x) = - factor*price*(power - alpha*curtailed_power) + stor1_e_cost*stor1_e_cap + stor1_p_cost*stor1_p_cap + stor2_e_cost*stor2_e_cap + stor2_p_cost*stor2_p_cap

    where factor = sum_n=1^(n_year) (1+discount_rate)**(-n).
    The objective function vector corresponds to the following design variables:

    *Formulation lp:*
        - Power from storage 1 in charge, shape-(n,)
        - Power from storage 1 in discharge, shape-(n,)
        - Power from storage 2 in charge, shape-(n,)
        - Power from storage 2 in discharge, shape-(n,)
        - Curtailed power, shape-(n,)
        - State of charge (energy) of storage 1, shape-(n+1,)
        - State of charge (energy) of storage 2, shape-(n+1,)
        - Power capacity of storage 1, shape-(1,)
        - Energy capacity of storage 1, shape-(1,)
        - Power capacity of storage 2, shape-(1,)
        - Energy capacity of storage 2, shape-(1,)

    The number of design variables is n_x = 7*n+6

    *Formulation milp:*

    Same as the lp formulation with the addition of:
        - Integer variables for storage 1, shape-(n,)
        - Integer variables for storage 2, shape-(n,)
    
    The number of design variables is n_x = 9*n+6
        
    *Formulation lp_alt:*
        - Power from storage 1, shape-(n,)
        - Power from storage 2, shape-(n,)
        - Curtailed power, shape-(n,)
        - State of charge (energy) of storage 1, shape-(n+1,)
        - State of charge (energy) of storage 2, shape-(n+1,)
        - Power capacity of storage 1, shape-(1,)
        - Energy capacity of storage 1, shape-(1,)
        - Power capacity of storage 2, shape-(1,)
        - Energy capacity of storage 2, shape-(1,)
        
    The number of design variables is n_x = 5*n+6

    Args:
        price (np.array): An array of electricity price to calculate the revenue [currency/MWh]
        n (int): the number of time steps [-]
        stor1_p_cost (float): cost of storage 1 per power capacity [currency/MW]
        stor1_e_cost (float): cost of storage 1 per energy capacity [currency/MWh]
        stor2_p_cost (float): cost of storage 2 per power capacity [currency/MW]
        stor2_e_cost (float): cost of storage 2 per energy capacity [currency/MWh]
        discount_rate (float): discount rate to calculate the NPV [-]
        n_year (int): number of years of operation of the project [-]
        options (dict): list of options for the problem formulation

            - formulation (str): Problem formulation for the storage model. Allowed values are 'lp', 'lp_alt', 'milp'. Default is lp_alt.
            - alpha_obj (float): penalty factor for the curtailed power in the objective function proportional to the price. Default is (1+1e-6)
            - beta_obj (float): penalty factor for the curtailed power in the objective function. Default is 0
            - epsilon (float): penalty factor to avoid simultaneous charge and discharge for the lp formulation. Default is 1e-3.

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
    
    # Default values for the options
    formulation = 'lp_alt'
    alpha_obj = DEFAULT_ALPHA_OBJ
    beta_obj = 0
    epsilon = 1e-3

    if options is not None:
        if 'formulation' in options.keys():
            formulation = options['formulation']
            assert (formulation == 'lp_alt') or (formulation == 'lp') or (formulation == 'milp')
        if 'epsilon' in options.keys():
            epsilon = options['epsilon']
            assert isinstance(epsilon, float)
        if 'alpha_obj' in options.keys():
            alpha_obj = options['alpha_obj']
            assert isinstance(alpha_obj, float)
        if 'beta_obj' in options.keys():
            beta_obj = options['beta_obj']
            assert isinstance(beta_obj, float)
    

    factor = npf.npv(discount_rate, np.ones(n_year))-1

    normed_price = 365 * 24 / n * np.reshape(price[:n], (n,1))*factor


    if formulation ==  'lp_alt':
        vec_obj = np.vstack((-normed_price ,             # Storage 1 power
                            -normed_price,             # Storage 2 power
                            alpha_obj*normed_price + beta_obj*np.ones((n,1)),             # Curtailed power
                            np.zeros((n+1, 1)),
                            np.zeros((n+1, 1)),
                            stor1_p_cost*np.ones((1,1)),           # minimize max batt power
                            stor1_e_cost*np.ones((1,1)),           # minimize max state of charge
                            stor2_p_cost*np.ones((1,1)),             # minimize max stor2_e power rate
                            stor2_e_cost*np.ones((1,1)))).squeeze()  # minimize max stor2_e energy capacity
    elif formulation ==  'lp':
        vec_obj = np.vstack((normed_price + epsilon*np.ones((n,1)),             # Storage 1 power charge
                            -normed_price + epsilon*np.ones((n,1)),             # Storage 1 power discharge
                            normed_price + epsilon*np.ones((n,1)),             # Storage 2 power charge
                            -normed_price + epsilon*np.ones((n,1)),             # Storage 2 power discharge
                            alpha_obj*normed_price + beta_obj*np.ones((n,1)),             # Curtailed power
                            np.zeros((n+1, 1)),
                            np.zeros((n+1, 1)),
                            stor1_p_cost*np.ones((1,1)),           # minimize max batt power
                            stor1_e_cost*np.ones((1,1)),           # minimize max state of charge
                            stor2_p_cost*np.ones((1,1)),             # minimize max stor2_e power rate
                            stor2_e_cost*np.ones((1,1)))).squeeze()  # minimize max stor2_e energy capacity
    elif formulation ==  'milp':
        vec_obj = np.vstack((normed_price + epsilon*np.ones((n,1)),             # Storage 1 power charge
                            -normed_price + epsilon*np.ones((n,1)),             # Storage 1 power discharge
                            normed_price + epsilon*np.ones((n,1)),             # Storage 2 power charge
                            -normed_price + epsilon*np.ones((n,1)),             # Storage 2 power discharge
                            alpha_obj*normed_price + beta_obj*np.ones((n,1)),             # Curtailed power
                            np.zeros((n+1, 1)),
                            np.zeros((n+1, 1)),
                            stor1_p_cost*np.ones((1,1)),           # minimize max batt power
                            stor1_e_cost*np.ones((1,1)),           # minimize max state of charge
                            stor2_p_cost*np.ones((1,1)),             # minimize max stor2_e power rate
                            stor2_e_cost*np.ones((1,1)),           # minimize max stor2_e energy capacity
                            np.zeros((n,1)),                          # integer variables storage 1
                            np.zeros((n,1)))).squeeze()              # integer variables storage 2
    

    return vec_obj

def build_lp_obj_revenues(price: np.ndarray, n: int, options: dict = None) -> np.ndarray:
    """Build the objective function vector for revenues maximization for the LP formulation.

    This function returns an objective vector corresponding to revenue maximization

        f(x) = -price*(power - alpha*curtailed_power)

    The objective function vector corresponds to the following design variables:
    
    *Formulation lp:*
        - Power from storage 1 in charge, shape-(n,)
        - Power from storage 1 in discharge, shape-(n,)
        - Power from storage 2 in charge, shape-(n,)
        - Power from storage 2 in discharge, shape-(n,)
        - Curtailed power, shape-(n,)
        - State of charge (energy) of storage 1, shape-(n+1,)
        - State of charge (energy) of storage 2, shape-(n+1,)
        - Power capacity of storage 1, shape-(1,)
        - Energy capacity of storage 1, shape-(1,)
        - Power capacity of storage 2, shape-(1,)
        - Energy capacity of storage 2, shape-(1,)

    The number of design variables is n_x = 7*n+6

    *Formulation milp:*

    Same as the lp formulation with the addition of:
        - Integer variables for storage 1, shape-(n,)
        - Integer variables for storage 2, shape-(n,)
    
    The number of design variables is n_x = 9*n+6

    *Formulation lp_alt:*
        - Power from storage 1, shape-(n,)
        - Power from storage 2, shape-(n,)
        - Curtailed power, shape-(n,)
        - State of charge (energy) of storage 1, shape-(n+1,)
        - State of charge (energy) of storage 2, shape-(n+1,)
        - Power capacity of storage 1, shape-(1,)
        - Energy capacity of storage 1, shape-(1,)
        - Power capacity of storage 2, shape-(1,)
        - Energy capacity of storage 2, shape-(1,)
        
    The number of design variables is n_x = 5*n+6

    Args:
        price (np.array): An array of electricity price to calculate the revenue [currency/MWh]
        n (int): the number of time steps [-]
        options (dict): list of options for the problem formulation

            - formulation (str): Problem formulation for the storage model. Allowed values are 'lp', 'lp_alt', 'milp'. Default is lp_alt.
            - alpha_obj (float): penalty factor for the curtailed power in the objective function proportional to the price. Default is (1+1e-6)
            - beta_obj (float): penalty factor for the curtailed power in the objective function. Default is 0
            - epsilon (float): penalty factor to avoid simultaneous charge and discharge for the lp formulation. Default is 1e-3.

    Returns:
        np.ndarray: A shape-(n_x,) array representing the objective function of the linear program [-]

    Raises:
        AssertionError: if the length of the price is below n, if any input is not finite
    """

    assert len(price) >= n
    assert np.all(np.isfinite(price))
    assert n != 0

    # Default values for the options
    formulation = 'lp_alt'
    alpha_obj = DEFAULT_ALPHA_OBJ
    beta_obj = 0
    epsilon = 1e-3

    if options is not None:
        if 'formulation' in options.keys():
            formulation = options['formulation']
            assert (formulation == 'lp_alt') or (formulation == 'lp') or (formulation == 'milp')
        if 'epsilon' in options.keys():
            epsilon = options['epsilon']
            assert isinstance(epsilon, float)
        if 'alpha_obj' in options.keys():
            alpha_obj = options['alpha_obj']
            assert isinstance(alpha_obj, float)
        if 'beta_obj' in options.keys():
            beta_obj = options['beta_obj']
            assert isinstance(beta_obj, float)

    if formulation ==  'lp_alt':
        vec_obj = np.vstack((-np.reshape(price[:n], (n,1)),             # Power from Storage 1
                            -np.reshape(price[:n], (n,1)),             # Power from Storage 2
                            np.reshape(price[:n], (n,1))*alpha_obj + np.ones((n,1))*beta_obj,   # Curtailed power
                            np.zeros((n+1, 1)),
                            np.zeros((n+1, 1)),
                            np.zeros((1,1)),           
                            np.zeros((1,1)),           
                            np.zeros((1,1)),             
                            np.zeros((1,1)))).squeeze()  
    
    elif formulation ==  'lp':
        vec_obj = np.vstack((np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 1 charge
                            -np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 1 discharge
                            np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 2 charge
                            -np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 2 discharge
                            np.reshape(price[:n], (n,1))*alpha_obj + np.ones((n,1))*beta_obj,   # Curtailed power
                            np.zeros((n+1, 1)),
                            np.zeros((n+1, 1)),
                            np.zeros((1,1)),           
                            np.zeros((1,1)),           
                            np.zeros((1,1)),             
                            np.zeros((1,1)))).squeeze()  
        
    elif formulation == 'milp':
        vec_obj = np.vstack((np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 1 charge
                            -np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 1 discharge
                            np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 2 charge
                            -np.reshape(price[:n], (n,1)) + epsilon*np.ones((n,1)),             # Power from Storage 2 discharge
                            np.reshape(price[:n], (n,1))*alpha_obj + np.ones((n,1))*beta_obj,   # Curtailed power
                            np.zeros((n+1, 1)),
                            np.zeros((n+1, 1)),
                            np.zeros((1,1)),           
                            np.zeros((1,1)),           
                            np.zeros((1,1)),             
                            np.zeros((1,1)),
                            np.zeros((n,1)),                        # integer variables storage 1
                            np.zeros((n,1)))).squeeze()              # integer variables storage 2

    return vec_obj

def build_lp_cst_sparse(power: np.ndarray, dt: float, p_min, p_max: float, n: int, stor1: Storage, stor2: Storage, stor1_p_cap_max: float = -1.0, stor2_p_cap_max: float = -1.0, stor1_e_cap_max: float = -1.0, stor2_e_cap_max: float = -1.0, options: dict = None ) -> tuple[sps.coo_matrix, np.ndarray, sps.coo_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Build the sparse constraints for the LP formulation of the dispatch optimization problem.

    Function to build the matrices and vectors for the constraints of the dispatch optimization problem, considering two different storage systems, and as a linear program. A sparse format is used to represent the matrices.

    The constraints are made of equality and inequality  constraints such that:

        - mat_eq * x = vec_eq
        - mat_ineq * x <= vec_ineq
        - bounds_lower <= x <= bounds_upper
  
    With n the number of time steps, the problem is made of:

        - n_x = 5*n+6 design variables (lp_alt formulation) or 7*n+6 (lp formulation) or 10*n+6 (milp formulation) 
        - n_eq = 0 equality constraints  (lp_alt formulation) or 2*n (lp and milp formulation)
        - n_ineq = 15*n+6 inequality constraints  (lp_alt formulation) or 6+11*n (lp formulation) or 6+17*n (milp formulation)

    The design variables for the linear problem are:

        - Power from storage 1, shape-(n,) (lp_alt formulation) or Power from storage 1 in charge, shape-(n,) and discharge, shape-(n,) (lp and milp formulation) 
        - Power from storage 2, shape-(n,) (lp_alt formulation) or Power from storage 2 in charge, shape-(n,) and discharge, shape-(n,) (lp and milp formulation) 
        - Curtailed power, shape-(n,)
        - State of charge (energy) of storage 1, shape-(n+1,)
        - State of charge (energy) of storage 2, shape-(n+1,)
        - Power capacity of storage 1, shape-(1,)
        - Energy capacity of storage 1, shape-(1,)
        - Power capacity of storage 2, shape-(1,)
        - Energy capacity of storage 2, shape-(1,)
        - Integer variables for storage 1, shape-(n,1) (milp formulation)
        - Integer variables for storage 2, shape-(n,1) (milp formulation)

    The equality constraints for the problem are:

        - Constraint to enforce the value of the first stored energy of storage 1 is equal to the last (size 1)
        - Constraint to enforce the value of the first stored energy of storage 2 is equal to the last (size 1)
        - (lp formulation only) Constraint to enforce the storage model for Storage 1 (size n)
        - (lp formulation only) Constraint to enforce the storage model for Storage 2 (size n)

    The inequality constraints for the problem are:

        - Constraints on the minimum and maximum combined power from production and storage assets (size 2*n)
        - (lp_alt formulation only) Constraints on the stored energy of storage 1 (size 2*n) 
        - (lp_alt formulation only) Constraints on the stored energy of storage 2 (size 2*n)
        - (lp_alt formulation only) Constraints on the maximmum and minimum power to and from storage 1 (size 2*n)
        - (lp_alt formulation only) Constraints on the maximmum and minimum power to and from storage 2 (size 2*n)
        - (lp formulation only) Constraints on the maximmum charge and discharge power for storage 1 (size 2*n)
        - (lp formulation only) Constraints on the maximmum charge and discharge power for storage 2 (size 2*n)
        - Constraints on the minimum and maximum stored energy in storage 1 (size 2*(n+1))
        - Constraints on the minimum and maximum stored energy in storage 2 (size 2*(n+1))
        - (milp formulation only) Constraint to enforce distinct charge and discharge in storage 1 (size 2*n)
        - (milp formulation only) Constraint to enforce distinct charge and discharge in storage 2 (size 2*n)
        - (milp formulation only) Constraint to enforce distinct curtailment and discharge in storage 1 (size n)
        - (milp formulation only) Constraint to enforce distinct curtailment and discharge in storage 1 (size n)
        - Constraint on the maximum combined storage power (size n)

    Args:
        power (np.ndarray): A shape-(n,) array for the power production from renewables [MW].
        dt (float): time step [s].
        p_min (float or np.ndarray): A float or shape-(n,) array for the minimum required power production [MW].
        p_max (float): maximum power production [MW]
        n (int): number of time steps [-].
        stor1 (Storage): object representing Storage 1 [-]. 
        stor2 (Storage): object representing Storage 2 [-].
        stor1_p_cap_max (float): maximum power capacity for storage 1 [MW]. Default to -1.0 when there is no limit in power rate.
        stor2_p_cap_max (float): maximum power capacity for storage 2 [MW]. Default to -1.0 when there is no limit in power rate.
        stor1_e_cap_max (float): maximum energy capacity for storage 1 [MWh]. Default to -1.0 when there is no limit.
        stor2_e_cap_max (float): maximum energy capacity for storage 2 [MWh]. Default to -1.0 when there is no limit.
        options (dict): list of options for the problem formulation

            - fixed_cap (bool): True if the storage capacity is fixed during the optimization. Default is False.
            - formulation (str): Problem formulation for the storage model. Allowed values are 'lp', 'lp_alt', 'milp'. Default is lp_alt.
    
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

    # Default values for the options
    formulation = 'lp_alt'
    fixed_cap = False

    if options is not None:
        if 'formulation' in options.keys():
            formulation = options['formulation']
            assert (formulation == 'lp_alt') or (formulation == 'lp') or (formulation == 'milp')
        if 'fixed_cap' in options.keys():
            fixed_cap = options['fixed_cap']
            assert isinstance(fixed_cap, bool)


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
    # power_ub =(np.array([ max(0, p_max - p) for p in  power[:n]])).reshape(n,1)


    # EQUALITY CONSTRAINTS
    # Constraint on first stored energy of storage 1
    if formulation == 'lp_alt':
        mat_stor1_first_e = sps.hstack((z_1n, z_1n, z_1n,
                                one_11, np.zeros((1, n-1)), -one_11,
                                z_1_np1,
                                z_11, z_11, z_11, z_11))
    elif formulation == 'lp' or formulation == 'milp':
        mat_stor1_first_e = sps.hstack((z_1n, z_1n, z_1n, z_1n, z_1n,
                                one_11, np.zeros((1, n-1)), -one_11,
                                z_1_np1,
                                z_11, z_11, z_11, z_11))
    
    vec_stor1_first_e = z_11

    # Constraint on first stored energy of storage 2
    if formulation == 'lp_alt':
        mat_stor2_first_e = sps.hstack((z_1n, z_1n, z_1n,
                                z_1_np1,
                                one_11, np.zeros((1, n-1)), -one_11,
                                z_11, z_11, z_11, z_11))
    elif formulation == 'lp' or formulation == 'milp':
        mat_stor2_first_e = sps.hstack((z_1n, z_1n, z_1n, z_1n, z_1n,
                                z_1_np1,
                                one_11, np.zeros((1, n-1)), -one_11,
                                z_11, z_11, z_11, z_11))
    vec_stor2_first_e = z_11

    # INEQ CONSTRAINT
    # Constraint on power production + storage 1 power + storage 2 power - curtailed_power >= p_min
    if formulation == 'lp_alt':
        mat_power_bound = sps.hstack((eye_n, eye_n, -eye_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    elif formulation == 'lp' or formulation == 'milp':
        mat_power_bound = sps.hstack((-eye_n, eye_n, -eye_n, eye_n, -eye_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    vec_power_min = p_min_vec - power[:n].reshape(n,1)
    vec_power_max = p_max - power[:n].reshape(n,1)

    # Constraint on the minimum and maximum stored energy of storage 1
    if formulation == 'lp_alt':
        mat_stor1_min_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, -eye_np1, z_np1,
                                z_np1_1, (1-stor1.dod)*one_np1_1, z_np1_1, z_np1_1))
        mat_stor1_max_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, eye_np1, z_np1,
                                z_np1_1, -1*one_np1_1, z_np1_1, z_np1_1))
    elif formulation == 'lp' or formulation == 'milp':
        mat_stor1_min_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n, z_np1_n, -eye_np1, z_np1,
                                z_np1_1, (1-stor1.dod)*one_np1_1, z_np1_1, z_np1_1))
        mat_stor1_max_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n, z_np1_n, eye_np1, z_np1,
                                z_np1_1, -1*one_np1_1, z_np1_1, z_np1_1))
    vec_stor1_min_energy = z_np1_1
    vec_stor1_max_energy = z_np1_1

    # Constraints on the minimum and maximum power of storage 1
    if formulation == 'lp_alt':
        mat_stor1_max_power = sps.hstack((eye_n, z_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
        mat_stor1_min_power = sps.hstack(( -eye_n, z_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
        vec_stor1_max_power = z_n1
        vec_stor1_min_power = z_n1

    elif formulation == 'lp' or formulation == 'milp':
        mat_stor1_max_power_c = sps.hstack((eye_n, z_n, z_n, z_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
        mat_stor1_max_power_d = sps.hstack(( z_n, eye_n, z_n, z_n, z_n, z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))        
        mat_stor1_max_power = sps.vstack((mat_stor1_max_power_c,  mat_stor1_max_power_d))
        vec_stor1_max_power = sps.vstack((z_n1, z_n1))        

    # Constraint on the maximum stored energy of storage 2
    if formulation == 'lp_alt':
        mat_stor2_min_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1,  -eye_np1,
                                z_np1_1, z_np1_1, z_np1_1, (1-stor2.dod)*one_np1_1))
        mat_stor2_max_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1,  eye_np1,
                                z_np1_1, z_np1_1, z_np1_1, -1*one_np1_1,))
    elif formulation == 'lp' or formulation == 'milp':
        mat_stor2_min_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n, z_np1_n,  z_np1,  -eye_np1,
                                z_np1_1, z_np1_1, z_np1_1, (1-stor2.dod)*one_np1_1))
        mat_stor2_max_energy = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n, z_np1_n, z_np1,  eye_np1,
                                z_np1_1, z_np1_1, z_np1_1, -1*one_np1_1))
    vec_stor2_min_energy = z_np1_1
    vec_stor2_max_energy = z_np1_1

    # Constraint on minimum and maximum power of storage 2
    if formulation == 'lp_alt':
        mat_stor2_max_power = sps.hstack((z_n, eye_n, z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, -one_n1, z_n1))

        mat_stor2_min_power = sps.hstack((z_n, -eye_n, z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, -one_n1, z_n1))
        vec_stor2_max_power = z_n1
        vec_stor2_min_power = z_n1
    elif formulation == 'lp' or formulation == 'milp':
        mat_stor2_max_power_c = sps.hstack(( z_n, z_n, eye_n, z_n, z_n, z_n_np1, z_n_np1,
                                     z_n1, z_n1, -1*one_n1, z_n1))
        mat_stor2_max_power_d = sps.hstack(( z_n, z_n,  z_n, eye_n, z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, -1*one_n1, z_n1))
        
        mat_stor2_max_power = sps.vstack((mat_stor2_max_power_c,  mat_stor2_max_power_d))
        vec_stor2_max_power = sps.vstack((z_n1, z_n1))

    # Constraint representing the storage model, linking stored energy to power and including storage losses
    if formulation == 'lp_alt':
        # e_(n+1) - e_(n) <= - dt * eff_in * p_(n)  (in)
        # e_(n+1) - e_(n) <= - dt/eff_out * p_(n) (out)
        mat_stor1_model_in = sps.hstack((dt * eye_n*stor1.eff_in, z_n, z_n,
                            -mat_diag_soc, -mat_last_soc, z_n_np1,
                            z_n1, z_n1, z_n1, z_n1))
        mat_stor1_model_out = sps.hstack((dt/stor1.eff_out * eye_n, z_n, z_n,
                                -mat_diag_soc, -mat_last_soc, z_n_np1,
                                z_n1, z_n1, z_n1, z_n1))
        vec_stor1_model_in = z_n1
        vec_stor1_model_out = z_n1
    elif formulation == 'lp' or formulation == 'milp':
        # e_(n+1) - e_(n) = dt * eff_in * p^charge_(n) - dt/eff_out * p^discharge_(n) 
        mat_stor1_model = sps.hstack((-dt * eye_n * stor1.eff_in, dt/stor1.eff_out * eye_n, z_n, z_n, z_n,
                            -mat_diag_soc, -mat_last_soc, z_n_np1,
                            z_n1, z_n1, z_n1, z_n1))
        vec_stor1_model = z_n1

    # Constraint on storage 2 energy:
    if formulation == 'lp_alt':
        #  e_(n+1) - e_(n) <= - dt * p_(n)
        #  e_(n+1) - e_(n) <= - dt/eta * p_(n)
        mat_stor2_model_in = sps.hstack((z_n, dt * eye_n *stor2.eff_in, z_n,
                                    z_n_np1, -mat_diag_soc, -mat_last_soc,
                                    z_n1, z_n1, z_n1, z_n1))
        mat_stor2_model_out = sps.hstack((z_n, dt/stor2.eff_out * eye_n, z_n,
                                    z_n_np1, -mat_diag_soc, -mat_last_soc,
                                    z_n1, z_n1, z_n1, z_n1))
        vec_stor2_model_in = z_n1
        vec_stor2_model_out = z_n1
    elif formulation == 'lp' or formulation == 'milp':
        # e_(n+1) - e_(n) = dt * p^charge_(n) - dt/eta * p^discharge_(n) 
        mat_stor2_model = sps.hstack((z_n, z_n, -dt*stor2.eff_in * eye_n, dt/stor2.eff_out * eye_n, z_n,
                            z_n_np1, -mat_diag_soc, -mat_last_soc,
                            z_n1, z_n1, z_n1, z_n1))
        vec_stor2_model = z_n1

    # Constraint to enforce charge distinct from discharge, and discharge distinct from curtailement with integer variables
    if formulation == 'milp':
        # Charging power  p^charge <= bigM * p_max * z_i
        mat_stor1_power_c_int = sps.hstack((eye_n, z_n, z_n, z_n, z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1, -BIG_M*p_max*eye_n, z_n))
        # Charging power  p^discharge <= bigM * p_max * ( 1- z_i)
        mat_stor1_power_d_int = sps.hstack(( z_n, eye_n, z_n, z_n, z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1, BIG_M*p_max*eye_n, z_n))  
        # Charging power  p^charge <= bigM * p_max * z_i
        mat_stor2_power_c_int = sps.hstack((z_n, z_n, eye_n, z_n, z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1, z_n, -BIG_M*p_max*eye_n))
        # Charging power  p^discharge <= bigM * p_max * ( 1- z_i)
        mat_stor2_power_d_int = sps.hstack(( z_n, z_n, z_n, eye_n, z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1, z_n, BIG_M*p_max*eye_n))
        # Curtailed energy
        mat_p_curt_stor1_int = sps.hstack(( z_n, z_n, z_n, z_n, eye_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1,  -BIG_M*p_max*eye_n, z_n))
        mat_p_curt_stor2_int = sps.hstack(( z_n, z_n, z_n, z_n, eye_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1, z_n, -BIG_M*p_max*eye_n))
        
        vec_stor1_power_c_int = z_n1
        vec_stor1_power_d_int = BIG_M*p_max*one_n1
        vec_stor2_power_c_int = z_n1
        vec_stor2_power_d_int = BIG_M*p_max*one_n1
        vec_p_curt_stor1_int = z_n1
        vec_p_curt_stor2_int = z_n1

    # Constraint on the maximum storage power, to avoid simultaneous discharge and curtailment
    # p <= max(p_max - p_res, 0)
    # or p^discharge - p^charge <= max(p_max-p_res)
    if formulation == 'lp_alt':
        mat_max_combined_power = sps.hstack((eye_n, eye_n, z_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    elif formulation == 'lp' or formulation == 'milp':
        mat_max_combined_power = sps.hstack((-eye_n, eye_n, -eye_n, eye_n, z_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))

    vec_max_combined_power =  np.array([max(p_max - p, 0) for p in power]).reshape(n,1)


    ## Assemble matrices
    if formulation == 'lp_alt':
        # mat_eq = sps.vstack((mat_stor1_first_e,  mat_stor2_first_e))
        # vec_eq = sps.vstack((vec_stor1_first_e,  vec_stor2_first_e)).toarray().squeeze()
        mat_eq = None
        vec_eq = None

        mat_ineq = sps.vstack((mat_stor1_first_e,  mat_stor2_first_e,
                               -1*mat_power_bound, mat_power_bound,
                                    mat_stor1_model_in, mat_stor1_model_out,
                                    mat_stor2_model_in, mat_stor2_model_out,
                                    mat_stor1_min_energy, mat_stor1_max_energy, mat_stor2_max_power,
                                    mat_stor2_min_power, mat_stor1_max_power,
                                    mat_stor1_min_power, mat_stor2_min_energy, mat_stor2_max_energy,
                                    mat_max_combined_power))
        vec_ineq = sps.vstack((vec_stor1_first_e,  vec_stor2_first_e,
                               -1*vec_power_min, vec_power_max,
                                    vec_stor1_model_in, vec_stor1_model_out,
                                    vec_stor2_model_in, vec_stor2_model_out,
                                    vec_stor1_min_energy, vec_stor1_max_energy, vec_stor2_max_power,
                                    vec_stor2_min_power, vec_stor1_max_power,
                                    vec_stor1_min_power, vec_stor1_min_energy, vec_stor2_max_energy,
                                    vec_max_combined_power)).toarray().squeeze()
    elif formulation == 'lp' or formulation == 'milp':
        mat_eq_lp = sps.vstack((mat_stor1_model, mat_stor2_model))
        vec_eq = sps.vstack(( vec_stor1_model, vec_stor2_model)).toarray().squeeze()

        mat_ineq_lp = sps.vstack((mat_stor1_first_e,  mat_stor2_first_e,-1*mat_power_bound, mat_power_bound,
                                    mat_stor1_min_energy, mat_stor1_max_energy, mat_stor2_max_power,
                                    mat_stor1_max_power,
                                    mat_stor2_min_energy, mat_stor2_max_energy,
                                    mat_max_combined_power))
        vec_ineq_lp = sps.vstack((vec_stor1_first_e,  vec_stor2_first_e,-1*vec_power_min, vec_power_max,
                                    vec_stor1_min_energy, vec_stor1_max_energy, vec_stor2_max_power,
                                    vec_stor1_max_power,
                                    vec_stor2_min_energy, vec_stor2_max_energy,
                                    vec_max_combined_power))
        
        if formulation == 'milp':
            # The equality matrix for the MILP formulation is constructed based on the LP one, as follows:
            # mat_ineq = | mat_eq_lp  |  right_block_eq |
            #
            # Where the right_block is a zeros (6n+2 x 2n) matrix corresponding to the additional integer variables for the existing constraints
            right_block_eq = sps.coo_array((2*n, 2*n))
            mat_eq = sps.hstack((mat_eq_lp, right_block_eq))

            # The inequality matrix for the MILP formulation is constructed based on the LP one, as follows:
            # mat_ineq = | mat_ineq_lp  |  right_block |
            #            | --------------------------- |
            #            |        lower_block          |
            # Where the right_block is a zeros (6n+2 x 2n) matrix corresponding to the additional integer variables for the existing constraints, and the lower block (6*n x 9n+6) corresponds to the contraints specific to the integer variables

            mat_right_block = sps.coo_array((11*n+6, 2*n))
            mat_lower_block = sps.vstack((mat_stor1_power_c_int, mat_stor1_power_d_int, mat_stor2_power_c_int, mat_stor2_power_d_int, mat_p_curt_stor1_int, mat_p_curt_stor2_int))
            vec_lower_block = sps.vstack((vec_stor1_power_c_int, vec_stor1_power_d_int, vec_stor2_power_c_int, vec_stor2_power_d_int, vec_p_curt_stor1_int, vec_p_curt_stor2_int))

            mat_ineq = sps.vstack(( sps.hstack((mat_ineq_lp, mat_right_block)), mat_lower_block))
            vec_ineq = sps.vstack((vec_ineq_lp, vec_lower_block)).toarray().squeeze()


        
        else:
            mat_eq = mat_eq_lp
            
            mat_ineq = mat_ineq_lp
            vec_ineq = vec_ineq_lp.toarray().squeeze()




    
    # BOUNDS ON DESIGN VARIABLES
    if formulation == 'lp_alt':
        if fixed_cap == False:
            bounds_lower = sps.vstack((-stor1_p_cap_max * one_n1,
                                        -stor2_p_cap_max * one_n1,
                                        z_n1,
                                        z_np1_1,
                                        z_np1_1,
                                        z_11,
                                        z_11,
                                        z_11,
                                        z_11)).toarray().squeeze()
        else:
            bounds_lower = sps.vstack((-stor1_p_cap_max * one_n1,
                                        -stor2_p_cap_max * one_n1,
                                        z_n1,
                                        z_np1_1,
                                        z_np1_1,
                                        stor1_p_cap_max*one_11,
                                        stor1_e_cap_max*one_11,
                                        stor2_p_cap_max*one_11,
                                        stor2_e_cap_max*one_11)).toarray().squeeze()

        bounds_upper = sps.vstack(( stor1_p_cap_max * one_n1,
                                    stor2_p_cap_max * one_n1,
                                    power[:n].reshape(n,1),
                                    stor1_e_cap_max*one_np1_1,
                                    stor2_e_cap_max*one_np1_1,
                                    stor1_p_cap_max*one_11,
                                    stor1_e_cap_max*one_11,
                                    stor2_p_cap_max*one_11,
                                    stor2_e_cap_max*one_11)).toarray().squeeze()
    elif formulation == 'lp' or formulation == 'milp':
        if fixed_cap == False:
            bounds_lower_lp = sps.vstack((z_n1,
                                       z_n1,
                                       z_n1,
                                       z_n1,
                                       z_n1,
                                        z_np1_1,
                                        z_np1_1,
                                        z_11,
                                        z_11,
                                        z_11,
                                        z_11))
        else:
            bounds_lower_lp = sps.vstack((z_n1,
                                       z_n1,
                                       z_n1,
                                       z_n1,
                                       z_n1,
                                        z_np1_1,
                                        z_np1_1,
                                        stor1_p_cap_max*one_11,
                                        stor1_e_cap_max*one_11,
                                        stor2_p_cap_max*one_11,
                                        stor2_e_cap_max*one_11))

        bounds_upper_lp = sps.vstack(( stor1_p_cap_max * one_n1,
                                   stor1_p_cap_max * one_n1,
                                    stor2_p_cap_max * one_n1,
                                    stor2_p_cap_max * one_n1,
                                    power[:n].reshape(n,1),
                                    stor1_e_cap_max*one_np1_1,
                                    stor2_e_cap_max*one_np1_1,
                                    stor1_p_cap_max*one_11,
                                    stor1_e_cap_max*one_11,
                                    stor2_p_cap_max*one_11,
                                    stor2_e_cap_max*one_11))

        if formulation == 'milp':
            bounds_lower = sps.vstack((bounds_lower_lp, z_n1, z_n1)).toarray().squeeze()
            bounds_upper = sps.vstack((bounds_upper_lp, one_n1, one_n1)).toarray().squeeze()
        else:
            bounds_lower = bounds_lower_lp.toarray().squeeze()
            bounds_upper = bounds_upper_lp.toarray().squeeze()


    return mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper

def solve_lp_sparse(price_ts: TimeSeries, prod1: Production,
                    prod2: Production, stor1: Storage, stor2: Storage,
                    discount_rate: float, n_year: int,
                    p_min, p_max: float,
                    n: int, options: dict = None) -> OpSchedule:
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
        options (dict): list of options for the problem formulation
        
            - formulation (str): Problem formulation for the storage model. Allowed values are 'lp', 'lp_alt', 'milp'. Default is lp_alt.
            - fixed_cap (bool): True if the storage capacity is fixed during the optimization. Default is False.
            - alpha_obj (float): penalty factor for the curtailed power in the objective function proportional to the price. Default is (1+1e-6)
            - beta_obj (float): penalty factor for the curtailed power in the objective function. Default is 0
            - epsilon (float): penalty factor to avoid simultaneous charge and discharge for the lp formulation. Default is 1e-3.

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

    # Default values for the options
    formulation = 'lp_alt'
    fixed_cap = False

    if options is not None:
        if 'formulation' in options.keys():
            formulation = options['formulation']
            assert (formulation == 'lp_alt') or (formulation == 'lp') or (formulation == 'milp')
        if 'fixed_cap' in options.keys():
            fixed_cap = options['fixed_cap']
            assert isinstance(fixed_cap, bool)

    power_res = prod1.power.data[:n] + prod2.power.data[:n]


    # Build the vector representing the objective function. If the storage capacity is fixed, the objective function is revenues, else it is NPV. 
    if fixed_cap:
        vec_obj = build_lp_obj_revenues(price_ts.data, n, options)
    else:
        vec_obj = build_lp_obj_npv(price_ts.data, n, stor1.p_cost, stor1.e_cost, stor2.p_cost, stor2.e_cost, discount_rate, n_year, options)

    # Build the matrices and vectors representing the constraints of the problem  
    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper =  build_lp_cst_sparse(power_res, dt, p_min, p_max, n, stor1, stor2, stor1_p_cap_max = stor1.p_cap, stor2_p_cap_max = stor2.p_cap, stor1_e_cap_max = stor1.e_cap, stor2_e_cap_max= stor2.e_cap, options = options)

    n_var = bounds_upper.shape[0]
    if vec_eq is not None:
        n_cstr_eq = vec_eq.shape[0]
    else:
        n_cstr_eq = 0
    n_cstr_ineq = vec_ineq.shape[0]

    assert n_var == bounds_lower.shape[0]
    assert n_var == vec_obj.shape[0]

    # Create the variable indicating integer variables
    if formulation == 'lp_alt' or formulation == 'lp':
        integrality = np.zeros_like(vec_obj)
    elif formulation == 'milp':
        integrality = np.concatenate(( np.zeros((n_var - 2*n)), np.ones((2*n))))

    # Reformat the lower and upper bounds in a single variable
    bounds = []
    for x in range(0, n_var):
        bounds.append((bounds_lower[x], bounds_upper[x]))


    # Solve the problem using linprog
    try:
        res = linprog(vec_obj, A_ub= mat_ineq, b_ub = vec_ineq, A_eq=mat_eq, b_eq=vec_eq, bounds=bounds, integrality = integrality, method = 'highs')
        x = res.x
    except:
        traceback.print_exc()
        raise RuntimeError from None

    if res.status != 0:
        print(res.message)
        raise RuntimeError
    
    # Extract solution
    if formulation == 'lp_alt':
        stor1_p = x[0:n] 
        stor2_p = x[n:2*n]
        p_cur = x[2*n:3*n]
        stor1_e = x[3*n:4*n+1]
        stor2_e = x[4*n+1:5*n+2]
        stor1_p_cap = x[5*n+2]
        stor1_e_cap = x[5*n+3]
        stor2_p_cap = x[5*n+4]
        stor2_e_cap = x[5*n+5]

    elif formulation == 'lp' or formulation == 'milp':
        stor1_p_charge = x[0:n] 
        stor1_p_discharge = x[n:2*n] 
        stor2_p_charge = x[2*n:3*n]
        stor2_p_discharge = x[3*n:4*n]
        p_cur = x[4*n:5*n]
        stor1_e = x[5*n:6*n+1]
        stor2_e = x[6*n+1:7*n+2]
        stor1_p_cap = x[7*n+2]
        stor1_e_cap = x[7*n+3]
        stor2_p_cap = x[7*n+4]
        stor2_e_cap = x[7*n+5]

        stor1_p = [-c + d for c, d in zip(stor1_p_charge, stor1_p_discharge)]
        stor2_p = [-c + d for c, d in zip(stor2_p_charge, stor2_p_discharge)]

    power_res_new = []
    power_losses_stor1 = []
    power_losses_stor2 = []


    for i in range(n):
        # Compute the power produced by the first production asset, and remove curtailment
        power_res_new.append(power_res[i]- p_cur[i])
        
        # Calculate the losses in the solution
        power_losses_stor1.append(-(stor1_e[i+1] - stor1_e[i] + dt*stor1_p[i])/dt)
        power_losses_stor2.append(-(stor2_e[i+1] - stor2_e[i] + dt*stor2_p[i])/dt)

    stor1_res = Storage(e_cap = stor1_e_cap,
                            p_cap = stor1_p_cap,
                            eff_in = stor1.eff_in,
                            eff_out = stor1.eff_out,
                            p_cost = stor1.p_cost,
                            e_cost = stor1.e_cost,
                            dod = stor1.dod,
                            lifetime= stor1.lifetime,
                            opex_fix= stor1.opex_fix,
                            opex_var=stor1.opex_var)
    stor2_res = Storage(e_cap = stor2_e_cap,
                            p_cap = stor2_p_cap,
                            eff_in = stor2.eff_in,
                            eff_out = stor2.eff_out,
                            p_cost = stor2.p_cost,
                            e_cost = stor2.e_cost,
                            dod = stor2.dod,
                            lifetime= stor2.lifetime,
                            opex_fix= stor2.opex_fix,
                            opex_var=stor2.opex_var)

    
    p_curtail = prod2.power.data[:n] + prod1.power.data[:n] - np.array(power_res_new)
    # prod1_res = Production(power_ts = TimeSeries(p_curtail, dt), p_cost= prod1.p_cost)
    prod1_res = prod1.curtail(p_curtail)
    

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
    
    verifiedModel = os_res.check_losses(TOL, verbose=False)

    if not verifiedModel:
            os_res.check_losses(TOL, verbose=True)
            print('Error above tolerance in solve_lp_sparse', formulation)

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

def financial_metrics(production_list: list[Production], storage_list: list[Storage], production_p: list[TimeSeries],
                 storage_p: list[TimeSeries], p_max_hpp: float, p_cost_shared: float, price: TimeSeries, m:int, discount_rate: float, added_price: float = 0) -> tuple[float, float, float, float, list[float]]:
    """Calculate financial metrics for a given hybrid power plant
    
    Five metrics are calculated: LCOE, NPV, IRR, Total CAPEX, and the vector of cashflow during the project.
    
    Args:
        production_list (list[Production]): List of power generation components (wind farm, solar PV, etc.)
        storage_list (list[Storage]): List of storage system components
        production_p (list[TimeSeries]): List of power production time series, corresponding to the components in production_list
        storage_p (list[TimeSeries]): List of storage power time series, corresponding to the components in storage_list
        p_max_hpp (float): Grid connection capacity of the hybrid power plant (MW)
        p_cost_shared (float): Shared CAPEX of the hybrid power plant (e.g., balance of plant), proportional to the grid connection capacity (Currency/MW)
        price (TimeSeries): Time series of electricity price to calculate the revenue (currency/MWh)
        m (int): number of years of operation of the project (-))
        discount_rate (float): discount rate for the calculation of discounted cashflows (-)
        added_price (float, optional): Value of electricity price added to the price timeseries, similar to a subsidy premium (currency/MWh).  
    
    Returns:
        tuple[float, float, float, float, list[float]]: [lcoe, npv, irr, capex_tot, cashflow]

    Raises:
        AssertionError: if any argument is of incorrect type, if the length of production or storage objects does not match the list of power timeseries, if the time increment is inconsistent.
    
    
    
    """
    # Check validity of inputs
    assert isinstance(production_list, list) and all(isinstance(p, Production) for p in production_list), \
        "production_list must be a list of Production objects."
    assert isinstance(storage_list, list) and all(isinstance(s, Storage) for s in storage_list), \
        "storage_list must be a list of Storage objects."
    assert isinstance(production_p, list) and all(isinstance(ts, TimeSeries) for ts in production_p), \
        "production_p must be a list of TimeSeries objects."
    assert isinstance(storage_p, list) and all(isinstance(ts, TimeSeries) for ts in storage_p), \
        "storage_p must be a list of TimeSeries objects."
    assert isinstance(p_max_hpp, (int, float)) and (p_max_hpp > 0), "p_max_hpp must be a positive numeric value."
    assert isinstance(p_cost_shared, (int, float)) and (p_cost_shared >= 0), "p_cost_shared must be a positive numeric value."
    assert isinstance(price, TimeSeries), "price must be a TimeSeries object."
    assert isinstance(m, int) and (m>0), "m must be a positive integer."
    assert isinstance(discount_rate, (float, int)) and (0 <= discount_rate <=1), "discount_rate must be a numeric value between 0 and 1."
    if added_price is not None:
        assert isinstance(added_price, (float, int)) and (0 <= added_price), "added_price must be a positive numeric value."

    assert len(production_p) == len(production_list), "the number of production object in production_list must match the number of power timeseries in production_p"
    assert len(storage_p) == len(storage_list), "the number of storage object in storage_list must match the number of power timeseries in storage_p"

    dt = production_p[0].dt
    assert all(power.dt == dt for power in production_p+storage_p), "the time increment dt should be consistent accross timeseries in production_p and storage_p"

    # Calculate AEP, CAPEX, OPEX and Revenues for all production and storage objects
    aep_production = [  sum(prod.data)*dt for prod in production_p]
    aep_storage_discharge = [  sum(np.maximum(stor.data, 0))*dt for stor in storage_p] # Sum of energy in discharge only - for the calculation of the variable OPEX
    aep_storage_tot = [  sum(stor.data)*dt for stor in storage_p]

    capex_production =  [ prod.get_tot_costs() for prod in production_list ] 
    capex_storage =  [ stor.get_tot_costs() for stor in storage_list ] 
    capex_shared = p_max_hpp*p_cost_shared # Shared CAPEX of Balance of plant (BOP)

    opex_production =  [ prod.opex_fix * prod.p_max + prod.opex_var * aep_prod for prod, aep_prod in zip(production_list, aep_production) ] 
    opex_storage =  [ stor.opex_fix * stor.p_cap + stor.opex_var * aep_stor for stor, aep_stor in zip(storage_list, aep_storage_discharge) ] 

    revenues_production =  [ sum([(p) *(l + added_price) for p, l in zip(prod.data, price.data)]) for prod in production_p] 
    revenues_storage =  [sum([(p) *(l + added_price) for p, l in zip(stor.data, price.data)]) for stor in storage_p] 

    # Expenses are recorded in tuple objects, where the first element is the year where the expense occurs
    # The CAPEX for the production objects and for the balance of plant occur on year 0.    
    capex_tuples = [(0, capex_shared)] + [(0, capex_prod) for capex_prod in capex_production]

    # The OPEX of production objects occurs every year except year 0.
    opex_tuples  = [(i, opex_prod) for i in range(1, m+1) for opex_prod in opex_production]

    # The CAPEX and OPEX for the storage systems takes into account the year of replacement
    # Here, we assume that the storage is not replaced if its lifetime is longer than the remaining lifetime of the project. For example, for a 25 year project, a storage system with a 8 year lifetime is replaced 3 times.
    capex_batt_tuples = []
    opex_batt_tuples = []
    for stor, capex, opex in zip(storage_list, capex_storage, opex_storage): 
        for i in range(m//stor.lifetime): # Looping over the number of batteries during the project lifetime
            capex_batt_tuples.append((i*stor.lifetime, capex)) # Adding CAPEX at each replacement year
            for j in range(stor.lifetime):
                opex_batt_tuples.append((i*stor.lifetime+j, opex)) # Adding OPEX at each year of operation
    
    # Combine all expense tuples into a single vector of costs indexed by year, where costs_vec[i] is the total expenses in year i
    costs_vec = [0 for _ in range(m+1)]
    for tuple in capex_tuples + opex_tuples + capex_batt_tuples + opex_batt_tuples:
        index_year = tuple[0]
        costs_vec[index_year] += tuple[1]

    # Represent the electricity production and revenues into a vector indexed by year    
    aep_vec = [sum(aep_production)+sum(aep_storage_tot) if i>0 else 0 for i in range(0, m+1)]
    revenues_vec = [sum(revenues_production)+ sum(revenues_storage) if i>0 else 0 for i in range(0, m+1)]
    
    # Combine costs and revenues into a vector of cashflow over the project lifetime
    cashflow = [(rev-cost) for rev, cost in zip(revenues_vec, costs_vec)]
    
    # Calculate the LCOE as the ratio between levelized costs and levelized AEP
    levelized_aep_bis = npf.npv(discount_rate, aep_vec)
    levelized_costs_bis = npf.npv(discount_rate, costs_vec)
    
    lcoe = levelized_costs_bis/levelized_aep_bis

    # Calculate NPV and IRR
    npv = npf.npv(discount_rate, [(rev-cost) for rev, cost in zip(revenues_vec, costs_vec)])

    irr = npf.irr([(rev-cost) for rev, cost in zip(revenues_vec, costs_vec)])
   
    # Calculate the total CAPEX, included the discounted cost of storage system replacement.
    capex_tot = sum(capex_production)+capex_shared

    for tuple in capex_batt_tuples:
        capex_tot+=tuple[1]/(1+discount_rate)**tuple[0]

    return lcoe, npv, irr, capex_tot, cashflow