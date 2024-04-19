"""This module defines kernel functions for sizing_opt_hpp.

The functions defined in this module are used to analyze or compute data
for the classes defined in sizing_opt_hpp.

Functions:
    power_calc: Calculate power production from wind speed.
    plot_xcorr: Plot cross-correlation between two signals.
    build_lp_cst: Build dense constraints for a LP. (depreciated)
    build_lp_obj_pareto: Build objective vector for a LP in a
        pareto formulation.
    build_lp_obj_npv: Build objective vector for NPV maximization.
    build_milp_obj: Build MILP objective function in a pareto way.
    build_milp_obj_npv: Build MILP objective function to minimize NPV.
    build_lp_cst_sparse: Build sparse constraints for a LP.
    build milp_cst_sparse: Build sparse constraints for a MILP.
    solve_lp: Build and solve a LP. (depreciated)
    linprog_mosek: Solve a LP using MOSEK.
    milp_mosek: Solve a MILP using MOSEK.
    solve_lp_sparse_pareto: Build and solve a LP with a pareto
        formulation.
    solve_lp_sparse_old: Build and solve a LP for NPV maximization.
        (depreciated)
    solve_lp_sparse: Build and solve a LP for NPV maximization.
    solve_milp_sparse: Build and solve a MILP.
"""

import sys
import traceback
import numpy as np
import numpy_financial as npf
from scipy.signal import correlate
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import mosek
import scipy.sparse as sps

from sizing_opt_hpp.components import Storage, OpSchedule, Production
from sizing_opt_hpp.timeseries import TimeSeries

def power_calc(wind: np.ndarray, radius: float, cp: float, v_in:float,
               v_r: float, v_out: float, p_max: float = None) -> np.ndarray:
    """Calculate power production from wind speed.

    Function to calculate the power output of wind based on the wind
    speed, the rotor radius, the power coefficient, the cut-in, rated
    and cut-out wind speed.

    Args:
        wind (np.ndarray): A shape-(n,) array of wind speeds [m/s].
        radius (float): wind turbine radius [m].
        cp (float): power coefficient [-].
        v_in, v_r, v_out (float): cut_in, rated, and cut_out wind speeds
            [m/s].
        p_max (float, optional): rated power [MW]. Default to None, but
            can be used to override the rated wind speed input.

    Returns:
        power (np.ndarray): A shape-(n,) array of power [MW].

    Raises:
        AssertionError: if argument wind is not an instance of np.array,
            if any of the input is a NaN, if the values are negative or
            if cp is above 1

    """
    rho = 1.225
    pi = 3.141591

    assert isinstance(wind, np.ndarray)
    assert np.all(np.isfinite(wind))
    assert np.isfinite(radius) and radius>=0
    assert np.isfinite(cp) and  0<=cp<=1
    assert np.isfinite(v_in) and v_in>=0
    assert np.isfinite(v_r) and v_r>=0
    assert np.isfinite(v_out) and v_out>=0

    power = wind**3 * 0.5 * rho * pi * radius **2 * cp * 1e-6

    power[np.logical_or(wind<v_in,wind>v_out)] = 0
    if p_max is None:
        power[wind>= v_r] = v_r**3 * 0.5 * rho * pi * radius **2 * cp * 1e-6
    else:
        power[power>= p_max] = p_max
    return power

def plot_xcorr(x: list[float], y: list[float], delta_t: float = 1/24) -> None:
    """Plot cross-correlation between two signals.

    Params:
        x (list[float]): first signal.
        y (list[float]): second signal.
        delta_t (float, optional): time step [s]. Default to 1/24. Used
            to scale the x axis in the plot.
    """
    n_max = max(len(x), len(y))
    n_min = min(len(x), len(y))

    if n_max == len(y):
        lags = np.arange(-n_max + 1, n_min)
    else:
        lags = np.arange(-n_min + 1, n_max)
    c = correlate((x - np.mean(x)) / np.std(x), (y - np.mean(y))  / np.std(y), 'full')

    plt.plot(lags*delta_t, c / n_min)
    plt.xlim([delta_t, max(lags*delta_t)])
    plt.xscale('log')
    plt.xlabel('Time lag [days]')
    plt.ylabel('Cross-correlation coefficient [-]')

def build_lp_cst(power: np.ndarray, dt: float, p_min: float | np.ndarray,
                 p_max: float, n: int, losses_batt: float, losses_h2: float,
                 rate_batt: float = -1.0, rate_h2: float = -1.0,
                 max_soc: float = -1.0, max_h2: float = -1.0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray]:
    """Build dense constraints for a LP. (depreciated)

    Function to build the matrices and vectors corresponding to the
    linear program representing the scheduling design problem. A dense
    format is used to represent the matrices.

    The constraints are made of equality and inequality constraints such
    that:

        mat_eq * x = vec_eq
        mat_ineq * x <= vec_ineq
        bounds_lower <= x <= bounds_upper

    With n the number of time steps, the problem is made of:
        n_x = 7*n+4 design variables
        n_eq = 4*n+4 equality constraints
        n_ineq = 6*n+1 inequality constraints

    The design variables for the linear problem are:
        Available power production, shape-(n,)
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        Losses from battery (>0), shape-(n,)
        Losses from electrolyzer (>0), shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max state of charge (battery capacity), shape-(1,)
        Max hydrogen power capacity, shape-(1,)

    The equality constraints for the problem are:
        Constraints on the battery state of charge (size n+1)
        Constraints on the hydrogen storage levels (size n+1)
        Constraint to enforce the value of the first state of charge
            (size 1)
        Constraint to enforce the value of the first hydrogen level
            (size 1)
        Constraints to enforce the value of the battery losses (size n)
        Constraints to enforce the value of the hydrogen losses (size n)

    The inequality constraints for the problem are:
        Constraints on the minimum and maximum combined power from
            renewables and storage systems (size 2*n)
        Constraints on the maximum state of charge (size n+1)
        Constraints on the maxmimum and minimum power to and from the
            hydrogen storage system (size 2*n)
        Constraints on the maxmimum power from the battery (size n)

    Params:
        power (np.ndarray): A shape-(n,) array for the power production
            from renewables [MW].
        dt (float): time step [s].
        p_min (float or np.ndarray): A float or shape-(n,) array for the
             minimum required power production [MW].
        p_max (float): maximum power production [MW]
        n (int): number of time steps [-].
        losses_batt (float): represents the portion of power lost during
            charge and discharge of the battery [-]. The same efficiency
            is assumed for charge and discharge.
        losses_h2 (float): represents the portion of power lost during
            charge and discharge of the hydrogen storage system [-]. The
            same efficiency is assumed for charge and discharge.
        rate_batt (float, optional): maximum power rate for the battery
            [MW]. Default to -1 when there is no limit in power rate.
        rate_h2 (float): maximum power rate for the hydrogen storage
            system [MW]. Default to -1 when there is no limit in power
             rate.
        max_soc (float): maximum state of charge for the battery [MWh].
            Default to -1 when there is no limit in state of charge.
        max_soc (float): maximum hydrogen level for the hydrogen storage
            [MWh]. Default to -1 when there is no limit in storage.

    Returns:
        mat_eq (np.ndarray): A shape-(n_eq, n_x) array for the matrix of
            the equality constraints [-]
        vec_eq (np.ndarray): A shape-(n_eq,) array for the vector of the
             equality constraints [-]
        mat_ineq (np.ndarray): A shape-(n_ineq, n_x) array for the
            matrix of the inequality constraints [-]
        vec_ineq (np.ndarray): A shape-(n_ineq,) array for the vector of
             the inequality constraints [-]
        bounds_lower (np.ndarray): A shape-(n_x,) array for the lower
            bounds [-]
        bounds_upper (np.ndarray): A shape-(n_x,) array for the upper
            bounds [-]

    Raises:
        ValueError: if argument p_min is not a float or a list of floats
    """
    init_batt_charge = 0
    init_h2_levels = 0

    if rate_batt == -1:
        rate_batt = p_max

    if rate_h2 == -1:
        rate_fc = p_max
        rate_h2 = p_max
    else:
        rate_fc = rate_h2

    if max_soc == -1:
        max_soc = n*dt*rate_batt #MWh

    if max_h2 == -1:
        max_h2 = 100*1400*0.0333 #MWh  (1400kg H2 and 1kg is 33.3 kWh)

    if isinstance(p_min, (np.ndarray, list)):
        assert len(p_min) >= n
        p_min_vec = p_min[:n].reshape(n,1)
    elif isinstance(p_min, (float, int)):
        p_min_vec = p_min * np.ones((n,1))
    else:
        raise ValueError("Input p_min in build_lp_cost must be a float, int,\
                         list or numpy.array")

    z_n = np.zeros((n,n))
    z_np1 = np.zeros((n+1,n+1))
    z_n_np1 = np.zeros((n , n+1))
    z_np1_n = z_n_np1.transpose()
    z_1n = np.zeros((1,n))
    z_n1 = np.zeros((n, 1))
    z_11 = np.zeros((1 , 1))
    z_1_np1 = np.zeros((1 , n+1))
    z_np1_1 = z_1_np1.transpose()
    eye_n = np.eye(n)
    eye_np1 = np.eye(n+1)
    one_11 = np.ones((1  ,1))
    one_n1 = np.ones((n , 1))
    one_np1_1 = np.ones((n+1, 1))

    mat_last_soc = np.vstack((np.zeros((n-1,1)), -1*one_11))
    mat_diag_soc = eye_n -   np.diag(np.ones(n-1), 1)

    # EQUALITY CONSTRAINTS
    # Constraint on wind power + battery power + h2 power >= p_min
    mat_power_bound = np.hstack((eye_n, eye_n, eye_n, z_n, z_n,
                                    z_n_np1, z_n_np1, z_n1,
                                    z_n1))
    vec_power_min = p_min_vec
    vec_power_max = one_n1* p_max

    # Constraint on state of charge:
    # soc_(n) - soc_(n+1) - dt * (P_batt - losses) = 0

    mat_c2 = np.hstack((z_n, -dt * eye_n, z_n, dt * eye_n,  z_n,
                            mat_diag_soc, mat_last_soc,  z_n_np1, z_n1, z_n1))
    vec_b2 = z_n1

    # Constraint on hydrogen levels:  h2_(n) - h2_(n+1) - dt * (P_el - losses_h2 - losses_fc) >=0
    mat_h2_soc = np.hstack((z_n,  z_n, -dt * eye_n,  z_n, dt*eye_n,
                                z_n_np1,
                                mat_diag_soc, mat_last_soc, z_n1, z_n1))
    vec_h2_soc = z_n1
    # Constraint on first state of charge = last state of charge
    mat_c3 = np.hstack((z_1n, z_1n, z_1n, z_1n, z_1n, one_11,
                            z_1n, z_1_np1,
                            -init_batt_charge*one_11, z_11))
    vec_b3 = z_11

    # Constraint on first h2 level
    mat_c5 = np.hstack((z_1n, z_1n, z_1n, z_1n, z_1n, z_1_np1,
                            one_11, z_1n,
                            -init_h2_levels*one_11, z_11))
    vec_b5 = z_11

    # Constraint on the battery losses
    mat_losses_batt = np.hstack((z_n, losses_batt*eye_n,z_n, -1*eye_n,
                                    z_n, z_n_np1,
                                    z_n_np1, z_n1, z_n1))
    vec_losses_batt = z_n1
    # Constraint on the h2 losses
    mat_losses_h2 = np.hstack((z_n,z_n, losses_h2*eye_n,
                                    z_n, -1*eye_n,
                                    z_n_np1, z_n_np1, z_n1,
                                    z_n1))
    vec_losses_h2 = z_n1

    # INEQ CONSTRAINT
    # Constraint on the maximum state of charge (i.e battery capacity)
    mat_max_soc = np.hstack((z_np1_n, z_np1_n,
                                z_np1_n, z_np1_n,
                                z_np1_n, eye_np1,
                                z_np1, -1*one_np1_1,
                                z_np1_1))
    vec_max_soc = z_np1_1


    mat_max_batt = np.hstack((z_n, eye_n, z_n, z_n, z_n,
                                    z_n_np1, z_n_np1,
                                    -1 *one_n1, z_n1))
    vec_max_batt = z_n1



    # Constraint on electrolyzer / fuel cell maximum power output
    mat_h2_max_power = np.hstack((z_n, z_n, eye_n, z_n, z_n,
                                        z_n_np1, z_n_np1,
                                        z_n1, -one_n1))
    vec_h2_max_power = z_n1
    mat_h2_min_power = np.hstack((z_n, z_n, -eye_n, z_n, z_n,
                                        z_n_np1, z_n_np1,
                                        z_n1, -one_n1))
    vec_h2_min_power = z_n1



    mat_eq = np.vstack((mat_c2, mat_c3,  mat_c5, mat_losses_batt,
                            mat_losses_h2, mat_h2_soc))
    vec_eq = np.vstack((vec_b2, vec_b3,  vec_b5, vec_losses_batt,
                            vec_losses_h2, vec_h2_soc)).squeeze()


    mat_ineq = np.vstack((-1*mat_power_bound, mat_power_bound,
                                mat_max_soc, mat_h2_max_power,
                                mat_h2_min_power, mat_max_batt))
    vec_ineq = np.vstack((-1*vec_power_min, vec_power_max,
                                vec_max_soc, vec_h2_max_power,
                                vec_h2_min_power, vec_max_batt)).squeeze()
    # BOUNDS ON DESIGN VARIABLES
    bounds_lower = np.vstack((z_n1,
                                -rate_batt * one_n1,
                                -rate_h2 * one_n1,
                                -rate_h2*one_n1,
                                -rate_h2*one_n1,
                                z_np1_1,
                                z_np1_1,
                                z_11,
                                z_11)).squeeze()

    bounds_upper = np.vstack((power[0:n].reshape(n,1),
                                rate_batt * one_n1,
                                rate_fc * one_n1,
                                rate_h2*one_n1,
                                rate_h2*one_n1,
                                max_soc*one_np1_1,
                                max_h2*one_np1_1,
                                max_soc*one_11,
                                rate_h2 * one_11)).squeeze()

    return mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper

def build_lp_obj_pareto(power: np.ndarray, price: np.ndarray, n: int,
                        eta: float, alpha: float) -> np.ndarray:
    """Build objective vector for a LP in a pareto formulation.

    This function returns an objective vector corresponding to 3
    objectives: maximize the annual revenue, minimizing the energy
    capacity of the battery and minimizing the power capacity of the
    hydrogen storage. The relative importance of the 3 objectives is
    tuned using the parameters eta and alpha, such that:

        f(x) = -eta*normed_price*power + (1-eta)*alpha*e_cap_batt
            + (1-eta)*(1-alpha)*p_cap_h2

    The objective vector corresponds to the following design variables:
        Available power production, shape-(n,)
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        Losses from battery (>0), shape-(n,)
        Losses from electrolyzer (>0), shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max battery power capacity, shape-(1,)
        Max state of charge (battery energy capacity), shape-(1,)
        Max hydrogen system power capacity, shape-(1,)
        Max hydrogen system energy capacity, shape-(1,)

    The number of design variables is n_x = 7*n+6

    Params:
        power (np.ndarray): An array for the power production [MW]
        price (np.array): An array for the price of electricity to
            calculate the revenue [currency/MWh]
        n (int): the number of time steps [-]
        eta (float): pareto parameter associated to revenue vs storage
            size [-]
        alpha (float): pareto parameter associated to battery size vs
            hydrogen size [-]

    Returns:
        vec_obj (np.ndarray): A shape-(n_x,) array representing the
            objective function of the linear program [-]

    Raises:
        AssertionError if the length of the price and power array are
            below n, if the sum of price is zero, if alpha or eta are
            not between 0 and 1, and if any input is not finite
    """
    assert len(power) >= n
    assert len(price) >= n
    assert np.all(np.isfinite(power))
    assert np.all(np.isfinite(price))
    assert np.isfinite(alpha)
    assert np.isfinite(eta)
    assert 0 <= alpha <= 1
    assert 0 <= eta <= 1
    assert sum(price[:n]) != 0

    eps = 1e-4
    eps_h2 = 1e-6
    normed_price = np.reshape(price[:n],
                              (n,1))/(sum(price[:n])*np.median(power[:n]))
    vector_c = np.vstack((-eta*normed_price,             # Wind power
                      -eta*normed_price,             # Batt power
                      -eta*normed_price,             # Power from fuel cell / to electrolyzer
                      0.0*np.ones((n,1)),      # minimize losses from batteries
                      0.0*np.ones((n,1)),      #minimize losses from electrolizer
                      np.zeros((n+1, 1)),           # SoC
                      np.zeros((n+1, 1)),           # H2 levels
                      (1-eta)*alpha*eps*np.ones((1,1)),           # minimize max batt power
                      (1-eta)*alpha*np.ones((1,1)),           # minimize max state of charge
                      (1-eta)*(1-alpha)*np.ones((1,1)),  # minimize max h2 power rate
                      (1-eta)*(1-alpha)*eps_h2*np.ones((1,1)))).squeeze()

    return vector_c

def build_lp_obj_npv(price: np.ndarray, n: int, batt_p_cost: float,
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
        Available power production, shape-(n,)
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        Losses from battery (>0), shape-(n,)
        Losses from electrolyzer (>0), shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max battery power capacity, shape-(1,)
        Max state of charge (battery energy capacity), shape-(1,)
        Max hydrogen system power capacity, shape-(1,)
        Max hydrogen system energy capacity, shape-(1,)

    The number of design variables is n_x = 7*n+6

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

    vec_obj = np.vstack((-normed_price,            # Wind power
                        -normed_price,             # Batt power
                        -normed_price,             # Power from fuel cell / to electrolyzer
                        1e-3*np.ones((n, 1)),
                        1e-3*np.ones((n, 1)),
                        np.zeros((n+1, 1)),
                        np.zeros((n+1, 1)),
                        batt_p_cost*np.ones((1,1)),           # minimize max batt power
                        batt_e_cost*np.ones((1,1)),           # minimize max state of charge
                        h2_p_cost*np.ones((1,1)),             # minimize max h2 power rate
                        h2_e_cost*np.ones((1,1)))).squeeze()  # minimize max h2 energy capacity

    return vec_obj

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

def build_milp_obj(power: np.ndarray, price: np.ndarray, n: int, eta: float,
                   alpha: float) -> np.ndarray:
    """Build objective vector for a MILP in a pareto formulation.

    This function returns an objective vector corresponding to 3
    objectives: maximize the annual revenue, minimizing the energy
    capacity of the battery and minimizing the power capacity of the
    hydrogen storage (electrolyzer and fuel cell). The relative
    importance of the 3 objectives is tuned using the parameters eta and
    alpha, such that:

        f(x) = -eta*normed_price*power + (1-eta)*alpha*e_cap_batt
            + (1-eta)*(1-alpha)*0.5*(p_cap_h2+p_cap_fc)

    The objective vector corresponds to the following design variables:
        Available power production, shape-(n,)
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        Losses from battery (>0), shape-(n,)
        Losses from fuel cell (>0), shape-(n,)
        Losses from electrolyzer (>0), shape-(n,)
        Extra losses from electrolyzer at high power (>0), shape-(n,)
        State of battery discharging (1) or charging (0), binary, shape-(n,)
        State of electrolyzer off (0 or 1), binary, shape-(n,)
        State of electrolyzer standby (0 or 1), binary, shape-(n,)
        State of electrolyzer on (0 or 1), binary, shape-(n,)
        State of electrolyzer on high power (0 or 1), binary, shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max power capacity for the battery, shape-(1,)
        Max state of charge (battery capacity), shape-(1,)
        Max power rate from h2 electrolyzer, shape-(1,)
        Max power rate from h2 fuel cell, shape-(1,)
        Max energy capacity for the h2 system, shape-(1,)

    The number of design variables is n_x = 14*n+7

    Params:
        power (np.ndarray): An array for the power production [MW]
        price (np.array): An array for the price of electricity to
            calculate the revenue [currency/MWh]
        n (int): the number of time steps [-]
        eta (float): pareto parameter associated to revenue vs storage
            size [-]
        alpha (float): pareto parameter associated to battery size vs
            hydrogen size [-]

    Returns:
        vec_obj (np.ndarray): A shape-(n_x,) array representing the
            objective function of the linear program [-]

    Raises:
        AssertionError if the length of the price and power array are
            below n, if the sum of price is zero, if alpha or eta are
            not between 0 and 1, and if any input is not finite
    """
    assert len(power) >= n
    assert len(price) >= n
    assert np.all(np.isfinite(power))
    assert np.all(np.isfinite(price))
    assert np.isfinite(alpha)
    assert np.isfinite(eta)
    assert 0 <= alpha <= 1
    assert 0 <= eta <= 1
    assert sum(price[:n]) != 0

    normed_price = np.reshape(price[:n], (n,1))/(sum(price[:n])*np.median(power[:n]))


    vec_obj = np.vstack((-eta*normed_price,             # Wind power
                      -eta*normed_price,             # Batt power
                      -eta*normed_price,             # Power from fuel cell / to electrolyzer
                      np.zeros((n,1)),      # minimize losses from batteries
                      np.zeros((n,1)),      # losses from fuel cell
                      np.zeros((n,1)),      # losses from electrolizer
                      np.zeros((n,1)),      # losses from electrolizer
                      np.zeros((n,1)),      # binaries
                      np.zeros((n,1)),
                      np.zeros((n,1)),
                      np.zeros((n,1)),
                      np.zeros((n,1)),
                      np.zeros((n+1, 1)),   # SoC Battery
                      np.zeros((n+1, 1)),   # H2 levels
                      1e-5*np.ones((1,1)),
                      (1-eta)*alpha*np.ones((1,1)),           # minimize max state of charge
                      (1-eta)*(1-alpha)*0.5*np.ones((1,1)),
                      (1-eta)*(1-alpha)*0.5*np.ones((1,1)),    #  h2 power rate
                      1e-5*np.ones((1,1)))).squeeze() 


    return vec_obj


def build_milp_obj_npv(price: np.ndarray, n: int, batt_p_cost: float,
                     batt_e_cost: float, h2_p_cost: float, h2_e_cost: float,
                     discount_rate: float, n_year: int) -> np.ndarray:
    """Build objective vector for a MILP to minimize NPV

    This function returns an objective vector corresponding the
    maximization of Net Present Value (NPV):

        f(x) = - factor*price*power + price_e_cap_batt*e_cap_batt
               + price_p_cap_batt*p_cap_batt
               + price_e_cap_h2*e_cap_h2 + price_p_cap_h2*p_cap_h2

    where factor = sum_n=1^(n_year) (1+discount_rate)**(-n)

    The objective vector corresponds to the following design variables:
        Available power production, shape-(n,)
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        Losses from battery (>0), shape-(n,)
        Losses from fuel cell (>0), shape-(n,)
        Losses from electrolyzer (>0), shape-(n,)
        Extra losses from electrolyzer at high power (>0), shape-(n,)
        State of battery discharging (1) or charging (0), binary, shape-(n,)
        State of electrolyzer off (0 or 1), binary, shape-(n,)
        State of electrolyzer standby (0 or 1), binary, shape-(n,)
        State of electrolyzer on (0 or 1), binary, shape-(n,)
        State of electrolyzer on high power (0 or 1), binary, shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max power capacity for the battery, shape-(1,)
        Max state of charge (battery capacity), shape-(1,)
        Max power rate from h2 electrolyzer, shape-(1,)
        Max power rate from h2 fuel cell, shape-(1,)
        Max energy capacity for the h2 system, shape-(1,)

    The number of design variables is n_x = 14*n+7

    Params:
        power (np.ndarray): An array for the power production [MW]
        price (np.array): An array for the price of electricity to
            calculate the revenue [currency/MWh]
        n (int): the number of time steps [-]
        eta (float): pareto parameter associated to revenue vs storage
            size [-]
        alpha (float): pareto parameter associated to battery size vs
            hydrogen size [-]

    Returns:
        vec_obj (np.ndarray): A shape-(n_x,) array representing the
            objective function of the linear program [-]

    Raises:
        AssertionError if the length of the price and power array are
            below n, if the sum of price is zero, if alpha or eta are
            not between 0 and 1, and if any input is not finite
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

    vec_obj = np.vstack((-normed_price,            # Wind power
                        -normed_price,             # Batt power
                        -normed_price,             # Power from fuel cell / to electrolyzer
                        np.zeros((n,1)),      # minimize losses from batteries
                        np.zeros((n,1)),      # losses from fuel cell
                        np.zeros((n,1)),      # losses from electrolizer
                        np.zeros((n,1)),      # losses from electrolizer
                        np.zeros((n,1)),      # binaries
                        np.zeros((n,1)),
                        np.zeros((n,1)),
                        np.zeros((n,1)),
                        np.zeros((n,1)),
                        np.zeros((n+1, 1)),   # SoC Battery
                        np.zeros((n+1, 1)),   # H2 levels
                        batt_p_cost*np.ones((1,1)),           # minimize max batt power
                        batt_e_cost*np.ones((1,1)),           # minimize max state of charge
                        1*h2_p_cost*np.ones((1,1)),             # minimize max h2 power rate
                        0*h2_p_cost*np.ones((1,1)),             # minimize max h2 power rate # temporarily disabled
                        h2_e_cost*np.ones((1,1)))).squeeze()  # minimize max h2 energy capacity


    return vec_obj

def build_lp_cst_sparse(power: np.ndarray, dt: float, p_min: float|np.ndarray,
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
        n_x = 7*n+6 design variables
        n_eq = 2*n+2 equality constraints
        n_ineq = 10*n+2 inequality constraints

    The design variables for the linear problem are:
        Available power production, shape-(n,)
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        Losses from battery (>0), shape-(n,)
        Losses from electrolyzer (>0), shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max battery power capacity, shape-(1,)
        Max state of charge (battery energy capacity), shape-(1,)
        Max hydrogen system power capacity, shape-(1,)
        Max hydrogen system energy capacity, shape-(1,)

    The equality constraints for the problem are:
        Constraints on the battery state of charge (size n)
        Constraints on the hydrogen storage levels (size n)
        Constraint to enforce the value of the first state of charge
            is equal to the last (size 1)
        Constraint to enforce the value of the first hydrogen level
            is equal to the last (size 1)

    The inequality constraints for the problem are:
        Constraints on the minimum and maximum combined power from
            renewables and storage systems (size 2*n)
        Constraints on the maximum state of charge (size n+1)
        Constraints on the maxmimum and minimum power to and from the
            hydrogen storage system (size 2*n)
        Constraints on the maxmimum and minimum power to and from the
            battery (size 2*n)
        Constraints on the maximum hydrogen levels (size n+1)
        Constraints to enforce the value of the battery losses (size n)
        Constraints to enforce the value of the hydrogen losses (size n)

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

    # EQUALITY CONSTRAINTS
    # Constraint on wind power + battery power + h2 power >= p_min
    mat_power_bound = sps.hstack((eye_n, eye_n, eye_n, z_n, z_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    vec_power_min = p_min_vec
    vec_power_max = one_n1*p_max

    # Constraint on state of charge:
    # soc_(n) - soc_(n+1) - dt * (P_batt - losses) = 0

    mat_soc = sps.hstack((z_n, -dt * eye_n, z_n, -dt * eye_n,  z_n,
                            mat_diag_soc, mat_last_soc, z_n_np1,
                            z_n1, z_n1, z_n1, z_n1))
    vec_soc = z_n1

    # Constraint on hydrogen levels:  h2_(n) - h2_(n+1) - dt * (P_el - losses_h2 ) >=0
    mat_h2_soc = sps.hstack((z_n,  z_n, -dt * eye_n,  z_n, -dt*eye_n,
                                z_n_np1, mat_diag_soc, mat_last_soc,
                                z_n1, z_n1, z_n1, z_n1))
    vec_h2_soc = z_n1

    # Constraint on first state of charge
    mat_first_soc = sps.hstack((z_1n, z_1n, z_1n, z_1n, z_1n,
                            one_11, np.zeros((1, n-1)), -one_11,
                            z_1_np1,
                            z_11, z_11, z_11, z_11))
    vec_first_soc = z_11

    # Constraint on first h2 level
    mat_first_soc_h2 = sps.hstack((z_1n, z_1n, z_1n, z_1n, z_1n,
                            z_1_np1,
                            one_11, np.zeros((1, n-1)), -one_11,
                            z_11, z_11, z_11, z_11))
    vec_first_soc_h2 = z_11

    # INEQ CONSTRAINT
    # Constraint on the maximum state of charge (i.e battery capacity)
    mat_max_soc = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n,
                                z_np1_n, eye_np1, z_np1,
                                z_np1_1, -1*one_np1_1, z_np1_1, z_np1_1))
    vec_max_soc = z_np1_1


    mat_max_batt = sps.hstack((z_n, eye_n, z_n, z_n, z_n,
                                    z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
    vec_max_batt = z_n1

    mat_min_batt = sps.hstack((z_n, -eye_n, z_n, z_n, z_n,
                                    z_n_np1, z_n_np1,
                                    -1*one_n1, z_n1, z_n1, z_n1))
    vec_min_batt = z_n1



    # Constraint on electrolyzer / fuel cell maximum power output
    mat_h2_max_power = sps.hstack((z_n, z_n, eye_n, z_n, z_n,
                                        z_n_np1, z_n_np1,
                                        z_n1, z_n1, -one_n1, z_n1))
    vec_h2_max_power = z_n1
    mat_h2_min_power = sps.hstack((z_n, z_n, -eye_n, z_n, z_n,
                                        z_n_np1, z_n_np1,
                                        z_n1, z_n1, -one_n1, z_n1))
    vec_h2_min_power = z_n1

    mat_max_h2 = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n,
                                z_np1_n, z_np1,  eye_np1,
                                z_np1_1, z_np1_1, z_np1_1, -1*one_np1_1,))
    vec_max_h2 = z_np1_1

    # Constraint on the battery losses
    mat_eps_batt = sps.hstack((z_n, eps_batt/(1-eps_batt)*eye_n, z_n, -1*eye_n,
                                    z_n, z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    vec_eps_batt = z_n1
    # Constraint on the h2 losses
    mat_eps_h2 = sps.hstack((z_n,z_n, eps_h2/(1-eps_h2)*eye_n,
                                    z_n, -1*eye_n,
                                    z_n_np1, z_n_np1,
                                    z_n1, z_n1, z_n1, z_n1))
    vec_eps_h2 = z_n1

    ## Assemble matrices
    mat_eq = sps.vstack((mat_soc, mat_first_soc,  mat_first_soc_h2, mat_h2_soc))
    vec_eq = sps.vstack((vec_soc, vec_first_soc,  vec_first_soc_h2,
                         vec_h2_soc)).toarray().squeeze()

    mat_ineq = sps.vstack((-1*mat_power_bound, mat_power_bound,
                                mat_max_soc, mat_h2_max_power,
                                mat_h2_min_power, mat_max_batt,
                                mat_min_batt, mat_max_h2, mat_eps_batt,
                                mat_eps_h2))
    vec_ineq = sps.vstack((-1*vec_power_min, vec_power_max,
                                vec_max_soc, vec_h2_max_power,
                                vec_h2_min_power, vec_max_batt,
                                vec_min_batt, vec_max_h2, vec_eps_batt,
                                vec_eps_h2)).toarray().squeeze()
    # BOUNDS ON DESIGN VARIABLES
    power_lb = np.array([min(p, p_max) for p in power[:n]])
    bounds_lower = sps.vstack((z_n1,
                                -rate_batt * one_n1,
                                -rate_h2 * one_n1,
                                z_n1,
                                z_n1,
                                z_np1_1,
                                z_np1_1,
                                z_11,
                                z_11,
                                z_11,
                                z_11)).toarray().squeeze()

    bounds_upper = sps.vstack((power[:n].reshape(n,1),
                                rate_batt * one_n1,
                                rate_h2 * one_n1,
                                rate_batt*one_n1,
                                rate_h2*one_n1,
                                max_soc*one_np1_1,
                                max_h2*one_np1_1,
                                rate_batt*one_11,
                                max_soc*one_11,
                                rate_h2*one_11,
                                max_h2*one_11)).toarray().squeeze()


    return mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper

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

def build_milp_cst_sparse(power: np.ndarray, dt: float, p_min: float,
                          p_max: float, n: int, losses_batt_in: float,
                          losses_batt_out: float, losses_h2: float, 
                          losses_fc: float, rate_batt: float = -1, 
                          rate_h2: float = -1, max_soc: float = -1,
                          max_h2: float = -1) -> tuple[sps.coo_matrix,
                                                         np.ndarray,
                                                         sps.coo_matrix,
                                                         np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]:
    """Build sparse constraints for a MILP.

    Function to build the matrices and vectors corresponding to the
    mixed-integer linear program representing the scheduling design
    problem. A sparse format is used to represent the matrices.

    The constraints are made of equality and inequality constraints such
    that:

        mat_eq * x = vec_eq
        mat_ineq * x <= vec_ineq
        bounds_lower <= x <= bounds_upper

    With n the number of time steps, the problem is made of:
        n_x = 14*n+7 design variables
        n_eq = 3*n+2 equality constraints
        n_ineq = 25*n+2 inequality constraints

    The design variables for the mixed-integer linear problem are:
        Available power production, shape-(n,)
        Power from battery (charge/discharge), shape-(n,)
        Power from fuel cell (>0) or electrolyzer (<0), shape-(n,)
        Losses from battery (>0), shape-(n,)
        Losses from fuel cell (>0), shape-(n,)
        Losses from electrolyzer (>0), shape-(n,)
        Extra losses from electrolyzer at high power (>0), shape-(n,)
        State of battery discharging/charging (1/0), binary, shape-(n,)
        State of electrolyzer off, binary, shape-(n,)
        State of electrolyzer standby, binary, shape-(n,)
        State of electrolyzer on, binary, shape-(n,)
        State of electrolyzer on high power, binary, shape-(n,)
        State of charge from battery, shape-(n+1,)
        Hydrogen levels, shape-(n+1,)
        Max power capacity for the battery, shape-(1,)
        Max state of charge (battery capacity), shape-(1,)
        Max power rate from h2 electrolyzer, shape-(1,)
        Max power rate from h2 fuel cell, shape-(1,)
        Max hydrogen levels, shape-(1,)

    The equality constraints for the problem are:
        Constraints on the battery state of charge (size n)
        Constraints on the hydrogen storage levels (size n)
        Constraint to enforce the value of the first state of charge
            (size 1)
        Constraint to enforce the value of the first hydrogen level
            (size 1)
        Constraints to enforce the sum of electrolyzer states is equal
            to 1 (size n)

    The inequality constraints for the problem are:
        Constraints on the minimum and maximum combined power from
            renewables and storage systems (size 2*n)
        Constraints on the maximum state of charge (size 2*(n+1))
        Constraints on the maxmimum and minimum power to and from the
            hydrogen storage system (size 2*n)
        Constraints on the maxmimum and minimum power to and from the
            battery (size 2*n)
        Constraints on the battery losses (size 4*n)
        Constraint on the fuel cell losses (size 3*n)
        Constraints on the electrolyzer losses (size 3*n)
        Constraints on the electrolyzer losses at high power (size 3*n)
        Constraints on the bounds of the power to the electrolyzer
            depending on the binary state (size 2*n)
        Constraints on the bounds of the battery power depending on themax_soc
            binary state (size 2*n)

    Params:
        power (np.ndarray): A shape-(n,) array for the power production
            from renewables [MW].
        dt (float): time step [s].
        p_min (float or np.ndarray): A float or shape-(n,) array for the
             minimum required power production [MW].
        p_max (float): maximum power production [MW]
        n (int): number of time steps [-].
        losses_batt_in, losses_batt_out (float): represents the portion 
            of power lost during Charge and discharge of the battery [-]
        losses_h2 (float): represents the portion of power lost during
            charge and discharge of the hydrogen storage system [-]. The
            same efficiency is assumed for charge and discharge.
        rate_batt (float, optional): maximum power rate for the battery
            [MW]. Default to -1 when there is no limit in power rate.
        rate_h2 (float): maximum power rate for the hydrogen storage
            system [MW]. Default to -1 when there is no limit in power
             rate.
        max_soc (float): maximum state of charge for the battery [MWh].
            Default to -1 when there is no limit in state of charge.
        max_soc (float): maximum hydrogen level for the hydrogen storage
            [MWh]. Default to -1 when there is no limit in storage.

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
        integrality (np.ndarray): A shape-(n_x,) binary array stating if
            a given variable is integer (1) or not (0).

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
    assert np.isfinite(losses_batt_in)
    assert np.isfinite(losses_batt_out)
    assert np.isfinite(losses_h2)

    rate_h2_min = 0.0*p_max
    p_sb = 0.0*p_max  #standby power
    p_mid = 10*p_max  #power from which the efficiency of the electrolyzer is reduced
    tmp_slope = 1.0 #0.8
    tmp_cst = -(tmp_slope-1) * p_mid * dt

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
        max_h2 = n*dt*rate_h2 #100*1400*0.0333 #MWh  (1400kg H2 and 1kg is 33.3 kWh)

    if isinstance(p_min, (np.ndarray, list)):
        assert len(p_min) >= n
        p_min_vec = p_min[:n].reshape(n,1)
    elif isinstance(p_min, (float, int)):
        p_min_vec = p_min * np.ones((n,1))
    else:
        raise ValueError("Input p_min in build_lp_cost must be a float, int, list or numpy.array")

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
    mat_diag_soc = eye_n -   sps.diags(np.ones(n-1),1)

    big_m = 10*p_max

    eye_n_losses_batt_in = sps.coo_array(losses_batt_in*np.eye(n))
    eye_n_losses_batt_out = sps.coo_array(losses_batt_out/(1-losses_batt_out)*np.eye(n))
    eye_n_losses_fc = sps.coo_array(losses_fc/(1-losses_fc)*np.eye(n))
    eye_n_big_m = sps.coo_array(big_m*np.eye(n))
    eye_n_losses_h2 = sps.coo_array(losses_h2*np.eye(n))
    eye_n_p_mid = sps.coo_array(p_mid*np.eye(n))
    eye_n_p_sb = sps.coo_array(p_sb*np.eye(n))
    eye_n_rate_h2 = sps.coo_array(rate_h2*np.eye(n))
    eye_n_rate_h2_min = sps.coo_array(rate_h2_min*np.eye(n))
    eye_n_dt = sps.coo_array(dt*np.eye(n))

    ## EQUALITY CONSTRAINTS
    # Constraint on state of charge:
    #   soc_(n) - soc_(n+1) - dt * (P_batt + losses) = 0
    mat_soc_batt = sps.hstack((z_n, -eye_n_dt, z_n, -eye_n_dt, z_n, z_n, z_n,
                               z_n, z_n,  z_n, z_n, z_n, mat_diag_soc,
                               mat_last_soc,  z_n_np1, 
                               z_n1, z_n1, z_n1, z_n1,z_n1,))
    vec_soc_batt = z_n1

    # Constraint on hydrogen levels:
    #   h2_(n)-h2_(n+1) - dt*(P_el + los_fc + los_h2 + los_h2_plus) =0
    mat_soc_h2 = sps.hstack((z_n, z_n, -eye_n_dt, z_n, -eye_n_dt, -eye_n_dt,
                             -eye_n_dt, z_n, z_n, z_n, z_n, z_n, z_n_np1,
                             mat_diag_soc, mat_last_soc,
                             z_n1, z_n1, z_n1, z_n1,z_n1,))
    vec_soc_h2 = z_n1

    # Constraint on the first state of charge of the battery = last one
    mat_1st_soc_batt = sps.hstack((z_1n, z_1n, z_1n, z_1n, z_1n, z_1n, z_1n, z_1n,
                            z_1n, z_1n, z_1n, z_1n, 
                            one_11,  sps.coo_array((1,n-1)), -one_11, 
                            z_1_np1,
                            z_11, z_11, z_11, z_11, z_11))
    vec_1st_soc_batt = z_11

    # Constraint on the first h2 level = last one
    mat_1st_soc_h2 = sps.hstack((z_1n, z_1n, z_1n, z_1n, z_1n, z_1n, z_1n,
                                 z_1n, z_1n, z_1n, z_1n, z_1n, z_1_np1,
                                 one_11,  sps.coo_array((1,n-1)), -one_11,
                                 z_11, z_11, z_11, z_11, z_11))
    vec_1st_soc_h2 = z_11

    # Constraint on the states of the electrolyzer
    #   z_off + z_sb + z_on + z_plus = 1
    mat_op_states = sps.hstack((z_n, z_n, z_n, z_n, z_n, z_n, z_n,  z_n,
                                   eye_n, eye_n, eye_n, eye_n, z_n_np1,
                                   z_n_np1, 
                                   z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_op_states = one_n1

    ## INEQUALITY CONSTRAINTS
    # Constraint on the power to the grid:
    #   p_min <= p_res + p_batt + p_h2 <= p_max
    mat_power_bound = sps.hstack((eye_n, eye_n, eye_n, z_n,z_n, z_n,
                                  z_n, z_n, z_n,  z_n, z_n, z_n,
                                  z_n_np1, z_n_np1, 
                                  z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_power_min = p_min_vec
    vec_power_max = one_n1* p_max

    # Constraint on the maximum battery power 
    mat_batt_max_power = sps.hstack((z_n, eye_n, z_n, z_n,z_n, z_n, z_n,
                                   z_n, z_n, z_n,  z_n, z_n, z_n_np1,
                                   z_n_np1, 
                                   -one_n1, z_n1, z_n1,  z_n1,  z_n1,))
    vec_batt_max_power = z_n1
    mat_batt_min_power = sps.hstack((z_n, -eye_n, z_n, z_n,z_n, z_n, z_n,
                                   z_n, z_n, z_n, z_n, z_n, z_n_np1,
                                   z_n_np1, 
                                   -one_n1, z_n1, z_n1,  z_n1, z_n1))
    vec_batt_min_power = z_n1
    

    # Constraint on the maximum state of charge = battery capacity
    mat_max_soc = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n,
                              z_np1_n, z_np1_n, z_np1_n,z_np1_n,z_np1_n,
                              z_np1_n, z_np1_n, z_np1_n, eye_np1, z_np1, 
                              z_np1_1, -one_np1_1, z_np1_1, z_np1_1, z_np1_1))
    vec_max_soc = z_np1_1

    # Constraint on the maximum power output from electrolyzer / fuel cell
    mat_h2_max_power = sps.hstack((z_n, z_n, eye_n, z_n,z_n, z_n, z_n,
                                   z_n, z_n, z_n,  z_n, z_n, z_n_np1,
                                   z_n_np1, 
                                   z_n1, z_n1,  -one_n1, z_n1, z_n1,))
    vec_h2_max_power = z_n1
    mat_h2_min_power = sps.hstack((z_n, z_n, -eye_n, z_n,z_n, z_n, z_n,
                                   z_n, z_n, z_n, z_n, z_n, z_n_np1,
                                   z_n_np1, 
                                   z_n1, z_n1, -one_n1, z_n1, z_n1))
    vec_h2_min_power = z_n1

     # Constraint on the maximum state of charge for the hydrogen system
    mat_max_soc_h2 = sps.hstack((z_np1_n, z_np1_n, z_np1_n, z_np1_n,
                              z_np1_n, z_np1_n, z_np1_n,z_np1_n,z_np1_n,
                              z_np1_n, z_np1_n, z_np1_n, z_np1, eye_np1, 
                              z_np1_1, z_np1_1, z_np1_1, z_np1_1, -one_np1_1))
    vec_max_soc_h2 = z_np1_1

    # Constraint on the battery losses
    #   los_bat =  eps_bat*p_bat if p_bat>0 (if z_bat = 1)
    #   los_bat = -eps_bat*p_bat if p_bat<0 (if z_bat = 0)
    # This is modeled through 4 constraints:
    # (1&2)  -M*z_bat <= eps_bat * p_bat + los_bat <= M*z_bat
    # (3&4)  -M*(1-z_bat) <= -eps_bat*p_bat + los_bat <= M*(1-z_bat)
    mat_lbatt_charge_max = sps.hstack((z_n, eye_n_losses_batt_in, z_n, eye_n,
                                       z_n, z_n, z_n, -eye_n_big_m, z_n, z_n,
                                       z_n, z_n, z_n_np1, z_n_np1, 
                                       z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lbatt_charge_max = z_n1

    mat_lbatt_charge_min = sps.hstack((z_n, -eye_n_losses_batt_in, z_n, -eye_n,
                                       z_n, z_n, z_n, -eye_n_big_m, z_n, z_n,
                                       z_n, z_n, z_n_np1, z_n_np1, 
                                       z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lbatt_charge_min = z_n1

    #discharge
    mat_lbatt_dcharge_max = sps.hstack((z_n, -eye_n_losses_batt_out, z_n,
                                        eye_n, z_n, z_n, z_n, eye_n_big_m, z_n,
                                        z_n, z_n, z_n, z_n_np1, z_n_np1, 
                                        z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lbatt_dcharge_max = one_n1 * big_m

    mat_lbatt_dcharge_min = sps.hstack((z_n, eye_n_losses_batt_out, z_n,
                                        -eye_n, z_n, z_n, z_n, eye_n_big_m,
                                        z_n, z_n, z_n, z_n, z_n_np1, z_n_np1,
                                        z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lbatt_dcharge_min = one_n1 * big_m

    ## Constraint on the losses associated to the fuel cell
    #   los_fc = eps_fc*p_h2 if p_h2>0 (if z_off = 1)
    #   los_fc = 0           if p_h2<0 (if z_off = 0)
    # This is modeled through 3 constraints:
    # (1)    los_fc <= M*z_off
    # (2&3)  -M(1-z_off) <= -eps_fc*p_h2 + los_fc <= M(1-z_off)

    # losses_fc = 0 if state_off = 0
    mat_lfc_max_simple = sps.hstack((z_n, z_n, z_n,
                                     z_n, eye_n, z_n, z_n,
                                     z_n, -eye_n_big_m, z_n, z_n,
                                     z_n, z_n_np1, z_n_np1,
                                     z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lfc_max_simple = z_n1

    mat_lfc_max = sps.hstack((z_n, z_n, -eye_n_losses_fc,
                              z_n, eye_n, z_n, z_n,
                              z_n, eye_n_big_m, z_n, z_n,
                              z_n, z_n_np1, z_n_np1, 
                              z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lfc_max = sps.coo_array(big_m * np.ones((n,1)))

    mat_lfc_min = sps.hstack((z_n, z_n, eye_n_losses_fc,
                              z_n, -eye_n, z_n, z_n,
                              z_n, eye_n_big_m, z_n, z_n,
                              z_n, z_n_np1, z_n_np1,
                              z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lfc_min = sps.coo_array(big_m * np.ones((n,1)))

    ## Constraint on the losses associated to the electrolyzer
    # losses_h2 = 0 if state_off = 1
    #   los_h2 = -eps_h2*p_h2 if p_h2<0 (if z_off = 0)
    #   los_h2 = 0            if p_h2>0 (if z_off = 1)
    # This is modeled through 3 constraints:
    # (1)    los_h2 <= M*(1-z_off)
    # (2&3)  -M(z_off) <= eps_h2*p_h2 + los_h2 <= M(z_off)
    mat_lh2_max_simple = sps.hstack((z_n, z_n, z_n,
                                     z_n, z_n, eye_n, z_n,
                                     z_n, eye_n_big_m, z_n, z_n, z_n,
                                     z_n_np1, z_n_np1, 
                                     z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lh2_max_simple = big_m*one_n1

    mat_lh2_max = sps.hstack((z_n, z_n, eye_n_losses_h2,
                              z_n, z_n, eye_n, z_n,
                              z_n, -eye_n_big_m, z_n, z_n, z_n,
                              z_n_np1, z_n_np1, 
                              z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lh2_max = z_n1

    mat_lh2_min = sps.hstack((z_n, z_n, -eye_n_losses_h2,
                              z_n, z_n, -eye_n, z_n,
                              z_n, -eye_n_big_m, z_n, z_n, z_n,
                              z_n_np1, z_n_np1, 
                              z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lh2_min = z_n1

    ## Constraint on the losses associated to the electrolyzer at high
    # power (p_h2>p_mid)
    # The objective is to model a reduced efficiency of the hydrogen
    # production such that:
    #   /\h2 = slope*dt*(power_to_electrolyzer - losses) + cst
    # with power_to_electrolyzer = -p_h2
    # However, in the code, the increase in hydrogen is modeled as
    #   /\h2 = -dt*(p_h2 + los_h2 + los_pl)
    # Matching the equations gives the following expression for los_pl:
    #   los_pl = (slope-1)*(p_h2+los_h2) - cst/dt if p_h2>p_mid (z_pl=1)
    #   los_pl = 0                                if p_h2<p_mid (z_pl=0)
    # This is modeled through 3 constraints:
    # (1)    los_pl <= M*z_pl
    # (2&3)  -M(1-z_pl) <= dt*(1-slope)*p_h2 + dt*(1-slope)*los_h2
    #                                        + los_pl + cst <= M(1-z_pl)
    mat_lh2_plus_max_simple = sps.hstack((z_n, z_n, z_n,
                                          z_n, z_n, z_n, eye_n,
                                          z_n, z_n, z_n, z_n, -eye_n_big_m,
                                          z_n_np1, z_n_np1, 
                                          z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lh2_plus_max_simple = z_n1

    mat_lh2_plus_max = sps.hstack((z_n, z_n,
                                   sps.coo_array(-(tmp_slope-1)*np.eye(n)),
                                   z_n,z_n,
                                   sps.coo_array(-(tmp_slope-1)*np.eye(n)),
                                   eye_n,
                                   z_n, z_n, z_n, z_n,
                                   sps.coo_array(big_m/dt *np.eye(n)),
                                   z_n_np1, z_n_np1,
                                   z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lh2_plus_max = one_n1 * (big_m - tmp_cst)/dt

    mat_lh2_plus_min = sps.hstack((z_n, z_n,
                                   sps.coo_array((tmp_slope-1)*np.eye(n)),
                                   z_n,z_n,
                                   sps.coo_array((tmp_slope-1)*np.eye(n)),
                                   -eye_n, z_n, z_n, z_n, z_n,
                                   sps.coo_array(big_m/dt *np.eye(n)),
                                   z_n_np1, z_n_np1,
                                   z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_lh2_plus_min = one_n1 * (big_m + tmp_cst)/dt


    ## Constraint on the power to the electrolyzer
    # The bounds on the power depends on the state:
    #   -rate_h2 <= -p_h2 <= 0  if z_off = 1
    #   p_sb <= -p_h2 <= p_sb   if z_sb = 1
    #   rate_h2_min <= -p_h2 <= p_mid if z_on = 1
    #   p_mid <= -p_h2 <= rate_h2     if z_pl = 1
    # These equations are modeled using two constraints:
    # (1) -p_h2 <= z_sb*p_sb + z_on*p_mid + z_pl*rate_h2
    # (2)  p_h2 <= z_off*rate_h2 - z_sb*p_sb - z_on*rate_h2_min
    #                                                      - z_pl*p_mid
    mat_op_state_power_max = sps.hstack((z_n, z_n, -eye_n, z_n, z_n, z_n,
                                         z_n, z_n, z_n, -eye_n_p_sb,
                                         -eye_n_p_mid, -eye_n_rate_h2,
                                         z_n_np1, z_n_np1, 
                                         z_n1, z_n1, z_n1, z_n1,z_n1))
    vec_op_state_power_max = z_n1

    mat_op_state_power_min = sps.hstack((z_n, z_n, eye_n, z_n, z_n, z_n,
                                         z_n, z_n, -eye_n_rate_h2,
                                         eye_n_p_sb, eye_n_rate_h2_min,
                                         eye_n_p_mid, z_n_np1, z_n_np1,
                                         z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_op_state_power_min = z_n1

    ## Constraint on the states of the battery
    #   z_batt = 0 if p_batt < 0 (charge), z_batt = 1 else (discharge)
    # This is modeled with two constraints:
    # (1&2)  -M*(1-z_batt) <= p_batt <= M*z_batt
    #
    mat_op_batt1 = sps.hstack((z_n, eye_n, z_n,
                               z_n, z_n, z_n, z_n,
                               -eye_n_big_m, z_n, z_n, z_n, z_n,
                               z_n_np1, z_n_np1, 
                               z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_op_batt1 = z_n1

    mat_op_batt2 = sps.hstack((z_n, -eye_n, z_n, z_n, z_n, z_n, z_n,
                               eye_n_big_m, z_n, z_n, z_n, z_n, z_n_np1,
                               z_n_np1, 
                               z_n1, z_n1, z_n1, z_n1, z_n1))
    vec_op_batt2 = big_m * one_n1

    ## ASSEMBLING THE EQUALITY AND INEQUALITY MATRICES AND VECTORS
    mat_eq = sps.vstack(( mat_soc_batt, mat_1st_soc_batt,  mat_1st_soc_h2,
                         mat_soc_h2, mat_op_states))
    vec_eq = sps.vstack(( vec_soc_batt, vec_1st_soc_batt,  vec_1st_soc_h2,
                         vec_soc_h2, vec_op_states)).toarray().squeeze()

    mat_ineq = sps.vstack((-mat_power_bound,
                           mat_power_bound, mat_max_soc,
                           mat_op_state_power_max,
                           mat_op_state_power_min,
                           mat_lbatt_charge_max,
                           mat_lbatt_charge_min,
                           mat_lbatt_dcharge_max,
                           mat_lbatt_dcharge_min,
                           mat_lfc_max,
                           mat_lfc_min,
                           mat_lfc_max_simple,
                           mat_lh2_max,
                           mat_lh2_min,
                           mat_lh2_max_simple,
                           mat_lh2_plus_max,
                           mat_lh2_plus_min,
                           mat_lh2_plus_max_simple,
                           mat_h2_max_power,
                           mat_h2_min_power,
                           mat_op_batt1,
                           mat_op_batt2,
                           mat_batt_max_power,
                           mat_batt_min_power,
                           mat_max_soc_h2))

    vec_ineq = sps.vstack((-vec_power_min, vec_power_max,
                            vec_max_soc, vec_op_state_power_max,
                            vec_op_state_power_min,
                            vec_lbatt_charge_max, vec_lbatt_charge_min,
                            vec_lbatt_dcharge_max, vec_lbatt_dcharge_min,
                            vec_lfc_max, vec_lfc_min,
                            vec_lfc_max_simple,
                            vec_lh2_max, vec_lh2_min,
                            vec_lh2_max_simple,
                            vec_lh2_plus_max,
                            vec_lh2_plus_min,
                            vec_lh2_plus_max_simple,
                            vec_h2_max_power, vec_h2_min_power,
                            vec_op_batt1,
                            vec_op_batt2,
                            vec_batt_max_power,
                            vec_batt_min_power,
                            vec_max_soc_h2)).toarray().squeeze()
    # BOUNDS ON DESIGN VARIABLES
    bounds_lower = sps.vstack((z_n1,
                                -rate_batt * one_n1,
                                -rate_h2 * one_n1,
                                z_n1,   #losses batt
                                z_n1,
                                z_n1,
                                z_n1,
                                z_n1,
                                z_n1,
                                z_n1,
                                z_n1,
                                z_n1,
                                z_np1_1,
                                z_np1_1,
                                z_11,
                                z_11,
                                z_11,
                                z_11,
                                z_11)).toarray().squeeze()

    bounds_upper = sps.vstack((power[0:n].reshape(n,1),
                                rate_batt * one_n1,
                                rate_fc * one_n1,
                                rate_batt * one_n1,
                                rate_h2*one_n1,
                                rate_h2*one_n1,
                                rate_h2*one_n1,
                                one_n1,
                                one_n1,
                                one_n1 * 0,
                                one_n1,
                                one_n1 * 0,
                                max_soc*one_np1_1,
                                max_h2*one_np1_1,
                                rate_batt*one_11,
                                max_soc*one_11,
                                rate_h2*one_11,
                                rate_h2*one_11*0,  #temporarily, rate_fc = rate_h2 and therefore one of the variables is disabled
                                max_h2*one_11)).toarray().squeeze()

    # Build the integrality vector stating which variables are integers.
    integrality = np.zeros_like(bounds_lower)
    integrality[7*n:12*n] = 1

    return mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper,\
          integrality

def solve_lp(power_ts: TimeSeries, price_ts: TimeSeries, stor_batt: Storage,
             stor_h2: Storage, eta: float, alpha: float,
             p_min: float|np.ndarray, p_max: float, n: int) -> OpSchedule:
    """Build and solve a LP. (depreciated)

    This function builds and solves the hybrid sizing and operation
    problem as a linear program.

    Params:
        power_ts (TimeSeries): Time series of the power production from
            wind and solar [MW].
        price_ts (TimeSeries): Time series of the price of electricity
            on the day-ahead market [currency/MWh].
        stor_batt (Storage): Object describing the battery storage.
        stor_h2 (Storage): Object describing the hydrogen storage system.
        eta (float): Pareto parameter for the revenue vs storage costs [-].
        alpha (float): Pareto parameter for the battery vs hydrogen cost [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational
            schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match.
        RuntimeError: if the optimization algorithm fails to solve the
            problem.
    """
    assert power_ts.dt == price_ts.dt

    dt = power_ts.dt

    eps_batt = 1 - stor_batt.eff_in * stor_batt.eff_out
    eps_h2 = 1 - stor_h2.eff_in * stor_h2.eff_out

    vec_obj_tmp = build_lp_obj_pareto(power_ts.data, price_ts.data, n, eta,
                                      alpha)

    vec_obj = np.concatenate((vec_obj_tmp[:7*n+2], [vec_obj_tmp[7*n+3]],
                             [vec_obj_tmp[7*n+4]]))

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub = build_lp_cst(power_ts.data,
                                                dt,
                                                p_min, p_max, n,
                                                eps_batt,
                                                eps_h2,
                                                rate_batt = stor_batt.p_cap,
                                                rate_h2 = stor_h2.p_cap,
                                                max_soc = stor_batt.e_cap,
                                                max_h2 = stor_h2.e_cap)

    bounds = []
    for x in range(0, len(lb)):
        bounds.append((lb[x], ub[x]))

    res = linprog(vec_obj, A_ub= mat_ineq, b_ub = vec_ineq, A_eq=mat_eq,
                  b_eq=vec_eq, bounds=bounds, method = 'highs')

    # print(res.message)
    if res.success is not True:
        raise RuntimeError(res.message)

    power_res = res.x[0:n]
    power_batt = res.x[n:2*n]
    power_h2 = res.x[2*n:3*n]
    # power_losses = res.x[3*n:4*n]
    # power_eps_h2 = res.x[4*n:5*n]
    soc = res.x[5*n:6*n]
    h2 = res.x[6*n+1:7*n+1]
    # final_h2 = res.x[7*n+1]
    batt_capacity = res.x[7*n+2]
    max_h2_power = res.x[7*n+3]

    stor_batt_res = Storage(e_cap = batt_capacity,
                            p_cap = max(power_batt),
                            eff_in = 0,
                            eff_out = 1-eps_batt)
    stor_h2_res = Storage(e_cap = np.max(h2),
                            p_cap = max_h2_power,
                            eff_in = 0,
                            eff_out = 1-eps_h2)

    prod_res = Production(power_ts = TimeSeries(power_res, dt), p_cost=0)

    os_res = OpSchedule(production_list = [prod_res],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(power_res, dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc, dt),
                                     TimeSeries(h2, dt)])

    return os_res

def linprog_mosek(n_x: int, n_eq: int, n_ineq: int,
                  mat_eq: sps.coo_matrix, vec_eq: np.ndarray | list[float],
                  mat_ineq: sps.coo_matrix, vec_ineq: np.ndarray | list[float],
                  vec_obj: np.ndarray | list[float],
                  lower_bound: np.ndarray | list[float],
                  upper_bound: np.ndarray | list[float],
                  verbose: bool = False) -> list[float]:
    '''Solve a LP using MOSEK.

    This function is an interface calling MOSEK to solve a linear
    program. It is inspired by the linear program example 6.1 in the
    MOSEK tutorial for python:
    https://docs.mosek.com/latest/pythonapi/tutorial-lo-shared.html#

    The matrix inputs are expressed as sparse arrays (sps.coo_matrix).
    The vector inputs are np.ndarray by defaults, but a list[float]
    input should work as well.

    The constraints are made of equality and inequality constraints
    expressed in the following form:

        mat_eq * x = vec_eq
        mat_ineq * x <= vec_ineq
        lower_bound <= x <= upper_bound

    Params:
        n_x (int): Number of design variables.
        n_eq (int): Number of equality constraints.
        n_ineq (int): Number of inequality constraints.
        mat_eq (sps.coo_matrix): A shape-(n_eq,n_x) sparse array for the
            matrix of the equality constraints.
        vec_eq (np.ndarray or list[float]): A shape-(n_eq,) array for
            the vector of the equality constraints.
        mat_ineq (sps.coo_matrix): A shape-(n_ineq,n_x) sparse array for
            the matrix of the inequality constraints.
        vec_ineq (np.ndarray or list[float]): A shape-(n_ineq,) array
            for the vector of the inequality constraints.
        vec_obj (np.ndarray or list[float]): A shape-(n_x,) array for
            the vector of objective function.
        lower_bound, upper_bound (np.ndarray or list[float]):
            A shape-(n_x,) array for the lower and upper bounds of the
            design variables
        verbose (bool, optional): A boolean describing if the function
            should print messages in the optimization. Default to False.

    Returns:
        xx (list[float]): A shape-(n_x,) array for the optimal design.

    Raises:
        RuntimeError: if the optimal solution is not found.
        AssertionError: if the shape of the matrices and vector do not
            match the arguments n_x, n_eq and n_ineq.
        (Other exceptions can be raised by the MOSEK solver.)
    '''
    assert mat_eq.shape[0] == n_eq
    assert mat_eq.shape[1] == n_x
    assert vec_eq.shape[0] == n_eq
    assert mat_ineq.shape[0] == n_ineq
    assert mat_ineq.shape[1] == n_x
    assert vec_ineq.shape[0] == n_ineq
    assert vec_obj.shape[0] == n_x
    assert lower_bound.shape[0] == n_x
    assert upper_bound.shape[0] == n_x

    assert isinstance(mat_eq, sps.coo_matrix)
    assert isinstance(mat_ineq, sps.coo_matrix)

    inf = 0 #variable inf only used symbolically
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # Create a task object
    with mosek.Task() as task:
        if verbose:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

        # Bound keys for constraints
        bkc = []
        for i in range(n_eq):
            bkc.append(mosek.boundkey.fx)

        for i in range(n_ineq):
            bkc.append(mosek.boundkey.up)

        # Bound values for constraints
        blc = []
        buc = []
        for i in range(n_eq):
            blc.append(vec_eq[i])
            buc.append(vec_eq[i])

        for i in range(n_ineq):
            blc.append(-inf)
            buc.append(vec_ineq[i])

        # Bound keys for variables
        bkx = []
        blx = []
        bux = []
        for i in range(n_x):
            bkx.append(mosek.boundkey.ra)
            blx.append(lower_bound[i])
            bux.append(upper_bound[i])

        # Objective coefficients
        c = vec_obj

        # Below is the sparse representation of the A
        # matrix stored by column.
        mat_a = sps.vstack((mat_eq, mat_ineq))

        numvar = len(bkx)
        numcon = len(bkc)

        # Append 'numcon' empty constraints.
        # The constraints will initially have no bounds.
        task.appendcons(numcon)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(numvar)

        for j in range(numvar):
            # Set the linear term c_j in the objective.
            task.putcj(j, c[j])

            # Set the bounds on variable j
            # blx[j] <= x_j <= bux[j]
            task.putvarbound(j, bkx[j], blx[j], bux[j])

        task.putaijlist(subi = mat_a.row, subj=mat_a.col, valij = mat_a.data)

        # Set the bounds on constraints.
         # blc[i] <= constraint_i <= buc[i]
        for i in range(numcon):
            task.putconbound(i, bkc[i], blc[i], buc[i])

        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)

        # task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.free_simplex)
        # Solve the problem
        task.optimize()

        # Get status information about the solution
        solsta = task.getsolsta(mosek.soltype.bas)

        if solsta == mosek.solsta.optimal:
            xx = task.getxx(mosek.soltype.bas)

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
        else:
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            if (solsta == mosek.solsta.dual_infeas_cer or
                solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")
            raise RuntimeError

        return xx

def milp_mosek(n_x: int, n_eq: int, n_ineq: int,
                  mat_eq: sps.coo_matrix, vec_eq: np.ndarray | list[float],
                  mat_ineq: sps.coo_matrix, vec_ineq: np.ndarray | list[float],
                  vec_obj: np.ndarray | list[float],
                  lower_bound: np.ndarray | list[float],
                  upper_bound: np.ndarray | list[float],
                  integrality: np.ndarray | list[int],
                  init_point: np.ndarray | list[float] = None,
                  verbose: bool = False) -> list[float]:
    '''Solve a MILP using MOSEK.

    This function is an interface calling MOSEK to solve a mixed-integer
    linear program. It is inspired by the linear program example 6.13 in
    the MOSEK tutorial for python:
    https://docs.mosek.com/latest/pythonapi/tutorial-mio-shared.html

    The matrix inputs are expressed as sparse arrays (sps.coo_matrix).
    The vector inputs are np.ndarray by defaults, but a list[float]
    input should work as well.

    The constraints are made of equality and inequality constraints
    expressed in the following form:

        mat_eq * x = vec_eq
        mat_ineq * x <= vec_ineq
        lower_bound <= x <= upper_bound

    Params:
        n_x (int): Number of design variables.
        n_eq (int): Number of equality constraints.
        n_ineq (int): Number of inequality constraints.
        mat_eq (sps.coo_matrix): A shape-(n_eq,n_x) sparse array for the
            matrix of the equality constraints.
        vec_eq (np.ndarray or list[float]): A shape-(n_eq,) array for
            the vector of the equality constraints.
        mat_ineq (sps.coo_matrix): A shape-(n_ineq,n_x) sparse array for
            the matrix of the inequality constraints.
        vec_ineq (np.ndarray or list[float]): A shape-(n_ineq,) array
            for the vector of the inequality constraints.
        vec_obj (np.ndarray or list[float]): A shape-(n_x,) array for
            the vector of objective function.
        lower_bound, upper_bound (np.ndarray or list[float]):
            A shape-(n_x,) array for the lower and upper bounds of the
            design variables.
        integrality (np.ndarray or list[float]): A shape-(n_x,) array
            describing if a given design variable is an integer or not.
        init_point (np.ndarray or list[float], optional): A shape-(n_x,)
            array containing an initial feasible point for the problem.
            Default to None.
        verbose (bool, optional): A boolean describing if the function
            should print messages in the optimization. Default to False.

    Returns:
        xx (list[float]): A shape-(n_x,) array for the optimal design.

    Raises:
        RuntimeError: if the optimal solution is not found.
        AssertionError: if the shape of the matrices and vector do not
            match the arguments n_x, n_eq and n_ineq.
        (Other exceptions can be raised by the MOSEK solver.)
    '''
    assert mat_eq.shape[0] == n_eq
    assert mat_eq.shape[1] == n_x
    assert vec_eq.shape[0] == n_eq
    assert mat_ineq.shape[0] == n_ineq
    assert mat_ineq.shape[1] == n_x
    assert vec_ineq.shape[0] == n_ineq
    assert vec_obj.shape[0] == n_x
    assert lower_bound.shape[0] == n_x
    assert upper_bound.shape[0] == n_x
    assert integrality.shape[0] == n_x

    assert isinstance(mat_eq, sps.coo_matrix)
    assert isinstance(mat_ineq, sps.coo_matrix)

    inf = 0 #variable inf only used symbolically
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # Create a task object
    with mosek.Task() as task:
        if verbose:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)

        # Bound keys for constraints
        bkc = []
        for i in range(n_eq):
            bkc.append(mosek.boundkey.fx)

        for i in range(n_ineq):
            bkc.append(mosek.boundkey.up)

        # Bound values for constraints
        blc = []
        buc = []
        for i in range(n_eq):
            blc.append(vec_eq[i])
            buc.append(vec_eq[i])

        for i in range(n_ineq):
            blc.append(-inf)
            buc.append(vec_ineq[i])


        # Bound keys for variables
        bkx = []
        blx = []
        bux = []
        for i in range(n_x):
            bkx.append(mosek.boundkey.ra)
            blx.append(lower_bound[i])
            bux.append(upper_bound[i])

        # Objective coefficients
        c = vec_obj

        # Below is the sparse representation of the A
        # matrix stored by column.

        mat_a = sps.vstack((mat_eq, mat_ineq))

        numvar = len(bkx)
        numcon = len(bkc)

        # Append 'numcon' empty constraints.
        # The constraints will initially have no bounds.
        task.appendcons(numcon)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(numvar)

        for j in range(numvar):
            # Set the linear term c_j in the objective.
            task.putcj(j, c[j])

            # Set the bounds on variable j
            # blx[j] <= x_j <= bux[j]
            task.putvarbound(j, bkx[j], blx[j], bux[j])

        task.putaijlist(subi = mat_a.row, subj=mat_a.col, valij = mat_a.data)

        # Set the bounds on constraints.
         # blc[i] <= constraint_i <= buc[i]
        for i in range(numcon):
            task.putconbound(i, bkc[i], blc[i], buc[i])

        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)

        # Define variables to be integers
        for i in range(len(integrality)):
            if integrality[i] == 1:
                task.putvartype(i,mosek.variabletype.type_int)

        # Include initial point
        if init_point is not None:
            # print('Input initial feasible point')
            # task.putxx(mosek.soltype.itg, init_point)
            task.putxxslice(mosek.soltype.itg, 0, n_x, init_point)
            task.putintparam(mosek.iparam.mio_construct_sol, mosek.onoffkey.on)



        # Set max solution time
        task.putdouparam(mosek.dparam.mio_max_time, 6000.0)
        task.putdouparam(mosek.dparam.mio_tol_rel_gap, 1.0e-4)
        # task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.free_simplex)
        # Solve the problem
        task.optimize()

        if init_point is not None:
            constr = task.getintinf(mosek.iinfitem.mio_construct_solution)
            constrV = task.getdouinf(mosek.dinfitem.mio_construct_solution_obj)
            solp = task.getintinf(mosek.iinfitem.mio_initial_feasible_solution)
            if verbose:
                print(f"Construct solution utilization: {0}\nConstruct solution\
                       objective: {1:.3f}\n".format(constr, constrV))
                print('Initial feasible solution provided:', solp)



        prosta = task.getprosta(mosek.soltype.itg)
        solsta = task.getsolsta(mosek.soltype.itg)

        # Output a solution
        xx = task.getxx(mosek.soltype.itg)

        if solsta in [mosek.solsta.integer_optimal]:
            if verbose:
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)
        elif solsta == mosek.solsta.prim_feas:
            print("Feasible solution found.")
            task.solutionsummary(mosek.streamtype.msg)
            raise RuntimeError
        elif mosek.solsta.unknown:
            if prosta == mosek.prosta.prim_infeas_or_unbounded:
                print("Problem status Infeasible or unbounded.\n")
            elif prosta == mosek.prosta.prim_infeas:
                print("Problem status Infeasible.\n")
            elif prosta == mosek.prosta.unkown:
                print("Problem status unkown.\n")
            else:
                print("Other problem status.\n")
            task.solutionsummary(mosek.streamtype.msg)
            raise RuntimeError
        else:
            print("Other solution status")
            # Print solution summary if there is an error
            task.solutionsummary(mosek.streamtype.msg)
            raise RuntimeError

        return xx

def solve_lp_sparse_pareto(power_ts: TimeSeries, price_ts: TimeSeries,
                           stor_batt: Storage, stor_h2: Storage, eta: float,
                           alpha: float, p_min: np.ndarray | float,
                           p_max: float, n: int) -> OpSchedule:
    """Build and solve a LP using a pareto multi-objective function.

    This function builds and solves the hybrid sizing and operation
    problem as a linear program. The objective function combines 3
    objectives: maximizing revenue, minimizing battery energy capacity
    and minimizing hydrogen storage capacity.

    Params:
        power_ts (TimeSeries): Time series of the power production from
            wind and solar [MW].
        price_ts (TimeSeries): Time series of the price of electricity
            on the day-ahead market [currency/MWh].
        stor_batt (Storage): Object describing the battery storage.
        stor_h2 (Storage): Object describing the hydrogen storage.
        eta (float): Pareto parameter for the revenue vs storage costs [-].
        alpha (float): Pareto parameter for the battery vs hydrogen cost [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational
            schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match.
        RuntimeError: if the optimization algorithm fails to solve the
            problem.
    """
    assert power_ts.dt == price_ts.dt

    dt = power_ts.dt

    eps_batt = 1 - stor_batt.eff_in * stor_batt.eff_out
    eps_h2 = 1 - stor_h2.eff_in * stor_h2.eff_out

    vec_obj_array = build_lp_obj_pareto(power_ts.data, price_ts.data, n, eta, alpha)


    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper = \
        build_lp_cst_sparse(power_ts.data, dt, p_min, p_max, n, eps_batt,
                            eps_h2, rate_batt = stor_batt.p_cap,
                            rate_h2 = stor_h2.p_cap, max_soc = stor_batt.e_cap,
                            max_h2= stor_h2.e_cap)

    n_var = bounds_upper.shape[0]
    n_cstr_eq = vec_eq.shape[0]
    n_cstr_ineq = vec_ineq.shape[0]

    assert n_var == bounds_lower.shape[0]
    assert n_var == vec_obj_array.shape[0]

    try:
        x = linprog_mosek(n_var, n_cstr_eq, n_cstr_ineq, mat_eq, vec_eq,
                          mat_ineq, vec_ineq, vec_obj_array, bounds_lower,
                          bounds_upper)
    except mosek.Error as e:
        print("ERROR: ", str(e.errno))
        if e.msg is not None:
            print("\t", e.msg)
            raise RuntimeError from None
    except:
        #import traceback
        traceback.print_exc()
        raise RuntimeError from None


    power_wind = x[0:n]
    power_batt = x[n:2*n]
    power_h2 = x[2*n:3*n]
    # power_losses = x[3*n:4*n]
    # power_eps_h2 = x[4*n:5*n]
    soc = x[5*n:6*n+1]
    h2 = x[6*n+1:7*n+2]
    # final_h2 = x[7*n+1]
    batt_p_capacity = x[7*n+2]
    batt_e_capacity = x[7*n+3]
    h2_p_capacity = x[7*n+4]
    h2_e_capacity = x[7*n+5]

    stor_batt_res = Storage(e_cap = batt_e_capacity,
                            p_cap = batt_p_capacity,
                            eff_in = 1,
                            eff_out = 1-eps_batt,
                            p_cost = stor_batt.p_cost,
                            e_cost = stor_batt.e_cost)
    stor_h2_res = Storage(e_cap = h2_e_capacity,
                            p_cap = h2_p_capacity,
                            eff_in = 1,
                            eff_out = 1-eps_h2,
                            p_cost = stor_h2.p_cost,
                            e_cost = stor_h2.e_cost)

    prod_res = Production(power_ts = TimeSeries(power_wind, dt), p_cost=0)

    os_res = OpSchedule(production_list = [prod_res],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(power_wind, dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc, dt),
                                     TimeSeries(h2, dt)])

    return os_res

def solve_lp_sparse_old(power_ts: TimeSeries, price_ts: TimeSeries,
                        stor_batt: Storage, stor_h2: Storage,
                        discount_rate: float, n_year: int,
                        p_min: np.ndarray | float, p_max: float,
                        n: int) -> OpSchedule:
    """Build and solve a LP for NPV maximization.

    This function builds and solves the hybrid sizing and operation
    problem as a linear program. The objective is to minimize the Net
    Present Value of the plant. In this function, the input for the
    power production is only represented by a TimeSeries object, so
    there is no information on the type of power production (wind or
    solar), or the associated costs.

    Params:
        power_ts (TimeSeries): Time series of the power production from
            wind and solar [MW].
        price_ts (TimeSeries): Time series of the price of electricity
            on theday-ahead market [currency/MWh].
        stor_batt (Storage): Object describing the battery storage.
        stor_h2 (Storage): Object describing the hydrogen storage system.
        eta (float): Pareto parameter for the revenue vs storage costs [-].
        alpha (float): Pareto parameter for the battery vs hydrogen cost [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational
            schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match.
        RuntimeError: if the optimization algorithm fails to solve the
            problem.
    """
    assert power_ts.dt == price_ts.dt

    dt = power_ts.dt

    eps_batt = 1 - stor_batt.eff_in * stor_batt.eff_out
    eps_h2 = 1 - stor_h2.eff_in * stor_h2.eff_out

    vec_obj = build_lp_obj_npv(price_ts.data, n, stor_batt.p_cost,
                               stor_batt.e_cost, stor_h2.p_cost,
                               stor_h2.e_cost, discount_rate, n_year)


    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper = \
        build_lp_cst_sparse(power_ts.data, dt, p_min, p_max, n, eps_batt,
                            eps_h2, rate_batt = stor_batt.p_cap,
                            rate_h2 = stor_h2.p_cap, max_soc = stor_batt.e_cap,
                            max_h2= stor_h2.e_cap)

    n_var = bounds_upper.shape[0]
    n_cstr_eq = vec_eq.shape[0]
    n_cstr_ineq = vec_ineq.shape[0]

    assert n_var == bounds_lower.shape[0]
    assert n_var == len(vec_obj)

    try:
        x = linprog_mosek(n_var, n_cstr_eq, n_cstr_ineq, mat_eq, vec_eq,
                          mat_ineq, vec_ineq, vec_obj, bounds_lower,
                          bounds_upper)
    except mosek.Error as e:
        print("ERROR: ", str(e.errno))
        if e.msg is not None:
            print("\t", e.msg)
            raise RuntimeError from None
    except:
        #import traceback
        traceback.print_exc()
        raise RuntimeError from None


    power_wind = x[0:n]
    power_batt = x[n:2*n]
    power_h2 = x[2*n:3*n]
    # power_losses_bat = x[3*n:4*n]
    # power_losses_h2 = x[4*n:5*n]
    soc = x[5*n:6*n]
    h2 = x[6*n+1:7*n+1]
    # final_h2 = x[7*n+1]
    batt_p_capacity = x[7*n+2]
    batt_e_capacity = x[7*n+3]
    h2_p_capacity = x[7*n+4]
    h2_e_capacity = x[7*n+5]

    stor_batt_res = Storage(e_cap = batt_e_capacity,
                            p_cap = batt_p_capacity,
                            eff_in = 1,
                            eff_out = 1-eps_batt,
                            p_cost = stor_batt.p_cost,
                            e_cost = stor_batt.e_cost)
    stor_h2_res = Storage(e_cap = h2_e_capacity,
                            p_cap = h2_p_capacity,
                            eff_in = 1,
                            eff_out = 1-eps_h2,
                            p_cost = stor_h2.p_cost,
                            e_cost = stor_h2.e_cost)

    prod_res = Production(power_ts = TimeSeries(power_wind, dt), p_cost=0)

    os_res = OpSchedule(production_list = [prod_res],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(power_wind, dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc, dt),
                                     TimeSeries(h2, dt)])

    return os_res

def solve_lp_sparse(price_ts: TimeSeries, prod_wind: Production,
                    prod_pv: Production, stor_batt: Storage, stor_h2: Storage,
                    discount_rate: float, n_year: int,
                    p_min: float | np.ndarray, p_max: float,
                    n: int) -> OpSchedule:
    """Build and solve a LP for NPV maximization.

    This function builds and solves the hybrid sizing and operation
    problem as a linear program. The objective is to minimize the Net
    Present Value of the plant. In this function, the input for the
    power production represented by two Production objects, one for wind
    and one for solar.

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

    vec_obj = build_lp_obj_npv(price_ts.data, n, stor_batt.p_cost,
                               stor_batt.e_cost, stor_h2.p_cost,
                               stor_h2.e_cost, discount_rate, n_year)


    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds_lower, bounds_upper = \
        build_lp_cst_sparse(power_res, dt, p_min, p_max, n, eps_batt, eps_h2,
                            rate_batt = stor_batt.p_cap, rate_h2 = stor_h2.p_cap,
                            max_soc = stor_batt.e_cap, max_h2= stor_h2.e_cap)

    n_var = bounds_upper.shape[0]
    n_cstr_eq = vec_eq.shape[0]
    n_cstr_ineq = vec_ineq.shape[0]

    assert n_var == bounds_lower.shape[0]
    assert n_var == vec_obj.shape[0]

    try:
        x = linprog_mosek(n_var, n_cstr_eq, n_cstr_ineq, mat_eq, vec_eq, mat_ineq,
         vec_ineq, vec_obj, bounds_lower, bounds_upper)
    except mosek.Error as e:
        print("ERROR: ", str(e.errno))
        if e.msg is not None:
            print("\t", e.msg)
            raise RuntimeError from None
    except:
        #import traceback
        traceback.print_exc()
        # sys.exit(1)
        raise RuntimeError from None


    power_res_new = x[0:n]
    power_batt = x[n:2*n]
    power_h2 = x[2*n:3*n]
    power_losses_bat = x[3*n:4*n]
    power_losses_h2 = x[4*n:5*n]
    soc = x[5*n:6*n]
    h2 = x[6*n+1:7*n+1]
    # final_h2 = x[7*n+1]
    batt_p_capacity = x[7*n+2]
    batt_e_capacity = x[7*n+3]
    h2_p_capacity = x[7*n+4]
    h2_e_capacity = x[7*n+5]

    stor_batt_res = Storage(e_cap = batt_e_capacity,
                            p_cap = batt_p_capacity,
                            eff_in = 1,
                            eff_out = 1-eps_batt,
                            p_cost = stor_batt.p_cost,
                            e_cost = stor_batt.e_cost)
    stor_h2_res = Storage(e_cap = h2_e_capacity,
                            p_cap = h2_p_capacity,
                            eff_in = 1,
                            eff_out = 1-eps_h2,
                            p_cost = stor_h2.p_cost,
                            e_cost = stor_h2.e_cost)
    
    prod_wind_res = Production(power_ts = TimeSeries(np.array(power_res_new) 
                                                     - np.array(prod_pv.power.data[:n]), dt), 
                                                     p_cost= prod_wind.p_cost)

    os_res = OpSchedule(production_list = [prod_wind_res, prod_pv],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(prod_wind_res.power.data[:n], dt),
                                        TimeSeries(prod_pv.power.data[:n], dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc, dt),
                                     TimeSeries(h2, dt)],
                        price = price_ts.data[:n])

    os_res.get_npv_irr(discount_rate, n_year)

    os_res.losses = [np.array(power_losses_bat) ,  np.array(power_losses_h2)] 

    return os_res

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

    try:
        x = linprog_mosek(n_var, n_cstr_eq, n_cstr_ineq, mat_eq, vec_eq, mat_ineq,
         vec_ineq, vec_obj, bounds_lower, bounds_upper)
    except mosek.Error as e:
        print("ERROR: ", str(e.errno))
        if e.msg is not None:
            print("\t", e.msg)
            raise RuntimeError from None
    except:
        #import traceback
        traceback.print_exc()
        # sys.exit(1)
        raise RuntimeError from None


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


def solve_milp_sparse(power_ts: TimeSeries, price_ts: TimeSeries,
                      stor_batt: Storage, stor_h2: Storage, eta: float,
                      alpha: float, p_min: float | np.ndarray, p_max: float,
                      n: int, init_point: np.ndarray = None,
                      verbose: bool = False) -> OpSchedule:
    """Build and solve a MILP representing the hybrid sizing problem.

    This function builds and solves the hybrid sizing and operation
    problem as a mixed-integer linear program. The objective is to
    minimize the Net Present Value of the plant. In this function,
    the input for the power production is only represented by a
    TimeSeries object, so there is no information on the type of power
    production (wind or solar), or the associated costs.

    Params:
        power_ts (TimeSeries): Time series of the power production from
            wind and solar [MW].
        price_ts (TimeSeries): Time series of the price of electricity
            on theday-ahead market [currency/MWh].
        stor_batt (Storage): Object describing the battery storage.
        stor_h2 (Storage): Object describing the hydrogen storage system.
        eta (float): Pareto parameter for the revenue vs storage costs [-].
        alpha (float): Pareto parameter for the battery vs hydrogen cost [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.
        init_point (np.ndarray, optional): Initial feasible point for
            the optimization algorithm. Default to None.
        verbose (bool, optional): Boolean describing if the optimization
            algorithm should output information. Default to False.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational
            schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match.
        RuntimeError: if the optimization algorithm fails to solve the
            problem.

    """
    assert power_ts.dt == price_ts.dt

    dt = power_ts.dt

    #assuming eps_batt_in == eps_batt_out
    eps_batt_in = 1 - stor_batt.eff_in #losses parameters
    eps_batt_out = 1 - stor_batt.eff_out #losses parameters
    eps_h2 = 1 - stor_h2.eff_in
    eps_fc= 1 - stor_h2.eff_out


    vec_obj = build_milp_obj(power_ts.data, price_ts.data, n, eta, alpha)

    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub, integrality = \
        build_milp_cst_sparse(power_ts.data, dt,  p_min, p_max, n, eps_batt_in,
                              eps_batt_out, eps_h2, eps_fc, rate_batt = stor_batt.p_cap,
                            rate_h2 = stor_h2.p_cap, max_soc = stor_batt.e_cap,
                            max_h2= stor_h2.e_cap)

    n_var = ub.shape[0]
    n_cstr_eq = vec_eq.shape[0]
    n_cstr_ineq = vec_ineq.shape[0]

    try:
        x = milp_mosek(n_var, n_cstr_eq, n_cstr_ineq, mat_eq, vec_eq, mat_ineq,
                     vec_ineq, vec_obj, lb, ub, integrality,
                     init_point = init_point, verbose = verbose)
    except mosek.Error as e:
        print("ERROR: ", str(e.errno))
        if e.msg is not None:
            print("\t", e.msg)
            raise RuntimeError from None
    except:
        #import traceback
        traceback.print_exc()
        raise RuntimeError from None


    power_res = x[0:n]
    power_batt = x[n:2*n]
    power_h2 = x[2*n:3*n]
    losses_batt=  x[3*n:4*n]
    losses_fc = x[4*n:5*n]
    losses_h2 = x[5*n:6*n]
    # losses_h2_plus_vec=  x[6*n:7*n]
    # state_batt = x[7*n:8*n]
    # state_off = x[8*n:9*n]
    # state_sb=  x[9*n:10*n]
    # state_on=  x[10*n:11*n]
    # state_plus=  x[11*n:12*n]
    soc = x[12*n:13*n]
    h2 = x[13*n+1:14*n+1]
    # final_h2 = x[14*n+1]
    p_batt = x[14*n+2]
    batt_capacity = x[14*n+3]
    max_h2_power = x[14*n+4]
    max_h2_power_fc = x[14*n+5]
    h2_e_cap = x[14*n+6]

    stor_batt_res = Storage(e_cap = batt_capacity,
                            p_cap = p_batt,
                            eff_in = 1-eps_batt_in,
                            eff_out = 1-eps_batt_out,
                            p_cost = stor_batt.p_cost,
                            e_cost = stor_batt.e_cost)
    stor_h2_res = Storage(e_cap = h2_e_cap,
                            p_cap = max_h2_power,
                            eff_in = 1-eps_h2,
                            eff_out = 1-eps_fc,
                            p_cost = stor_h2.p_cost,
                            e_cost = stor_h2.e_cost)

    prod_res = Production(power_ts = TimeSeries(power_res, dt), p_cost=0)

    os_res = OpSchedule(production_list = [prod_res],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(power_res, dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc, dt),
                                     TimeSeries(h2, dt)],
                        price = price_ts.data[:n])

                                    #  TimeSeries(losses_batt, dt),
                                    #  TimeSeries(losses_h2, dt),
                                    #  TimeSeries(losses_fc, dt)
    os_res.losses = [np.array(losses_batt) ,  np.array(losses_h2)+np.array(losses_fc)] 

    return os_res

def solve_milp_sparse_npv(price_ts: TimeSeries, prod_wind: Production,
                          prod_pv: Production, stor_batt: Storage, 
                          stor_h2: Storage, discount_rate: float, n_year: int,
                          p_min: float | np.ndarray, p_max: float, n: int, 
                          init_point: np.ndarray = None,
                          verbose: bool = False) -> OpSchedule:
    """Build and solve a MILP for NPV maximization.

    This function builds and solves the hybrid sizing and operation
    problem as a mixed-integer linear program. The objective is to
    minimize the Net Present Value of the plant. In this function, the 
    input for the power production represented by two Production 
    objects, one for wind and one for solar.

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
        init_point (np.ndarray, optional): Initial feasible point for
            the optimization algorithm. Default to None.
        verbose (bool, optional): Boolean describing if the optimization
            algorithm should output information. Default to False.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational
            schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match.
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

    #assuming eps_batt_in == eps_batt_out
    eps_batt_in = 1 - stor_batt.eff_in #losses parameters
    eps_batt_out = 1 - stor_batt.eff_out
    eps_h2 = 1 - stor_h2.eff_in
    eps_fc = 1 - stor_h2.eff_out


    vec_obj = build_milp_obj_npv(price_ts.data, n, stor_batt.p_cost,
                               stor_batt.e_cost, stor_h2.p_cost,
                               stor_h2.e_cost, discount_rate, n_year)

    
    mat_eq, vec_eq, mat_ineq, vec_ineq, lb, ub, integrality = \
        build_milp_cst_sparse(power_res, dt,  p_min, p_max, n, eps_batt_in,
                              eps_batt_out, eps_h2, eps_fc, 
                              rate_batt = stor_batt.p_cap, 
                              rate_h2 = stor_h2.p_cap, 
                              max_soc = stor_batt.e_cap,
                              max_h2 = stor_h2.e_cap)

    n_var = ub.shape[0]
    n_cstr_eq = vec_eq.shape[0]
    n_cstr_ineq = vec_ineq.shape[0]

    try:
        x = milp_mosek(n_var, n_cstr_eq, n_cstr_ineq, mat_eq, vec_eq, mat_ineq,
                     vec_ineq, vec_obj, lb, ub, integrality,
                     init_point = init_point, verbose = verbose)
    except mosek.Error as e:
        print("ERROR: ", str(e.errno))
        if e.msg is not None:
            print("\t", e.msg)
            raise RuntimeError from None
    except:
        #import traceback
        traceback.print_exc()
        raise RuntimeError from None


    power_wind = x[0:n]
    power_batt = x[n:2*n]
    power_h2 = x[2*n:3*n]
    losses_bat =  x[3*n:4*n]
    losses_fc = x[4*n:5*n]
    losses_h2 = x[5*n:6*n]
    # losses_h2_plus_vec=  x[6*n:7*n]
    # state_batt = x[7*n:8*n]
    # state_off = x[8*n:9*n]
    # state_sb=  x[9*n:10*n]
    # state_on=  x[10*n:11*n]
    # state_plus=  x[11*n:12*n]
    soc = x[12*n:13*n]
    h2 = x[13*n+1:14*n+1]
    # final_h2 = x[14*n+1]
    p_batt = x[14*n+2]
    batt_capacity = x[14*n+3]
    max_h2_power = x[14*n+4]
    # max_h2_power_fc = x[14*n+5]
    h2_e_cap = x[14*n+6]

    stor_batt_res = Storage(e_cap = batt_capacity,
                            p_cap = p_batt,
                            eff_in = 1-eps_batt_in,
                            eff_out = 1-eps_batt_out,
                            p_cost = stor_batt.p_cost,
                            e_cost = stor_batt.e_cost)
    stor_h2_res = Storage(e_cap = h2_e_cap,
                            p_cap = max_h2_power,
                            eff_in = 1-eps_h2,
                            eff_out = 1-eps_fc,
                            p_cost = stor_h2.p_cost,
                            e_cost = stor_h2.e_cost)

    prod_wind_res = Production(TimeSeries(power_wind 
                                          - prod_pv.power.data[:n], dt), 
                                          p_cost = prod_wind.p_cost)

    os_res = OpSchedule(production_list = [prod_wind_res, prod_pv],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(prod_wind_res.power.data[:n], dt),
                                        TimeSeries(prod_pv.power.data[:n], dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc, dt),
                                     TimeSeries(h2, dt)],
                        price = price_ts.data[:n])

                                    #  TimeSeries(losses_batt, dt),
                                    #  TimeSeries(losses_h2, dt),
                                    #  TimeSeries(losses_fc, dt)
    os_res.losses = [np.array(losses_bat) ,  np.array(losses_fc)+np.array(losses_h2)] 
    # os_res.losses[0] = losses_bat
    # os_res.losses_h2 = losses_h2
    # os_res.losses_fc = losses_fc

    os_res.get_npv_irr(discount_rate, n_year)

    return os_res


def os_rule_based(price_ts: TimeSeries, prod_wind: Production,
                  prod_pv: Production, stor_batt: Storage, stor_h2: Storage,
                  discount_rate: float, n_year: int, p_min: float | np.ndarray,
                  p_rule: float, price_min: float,
                  n: int) -> OpSchedule:

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
    import rainflow
    soc_h2_max = max(soc_h2)
    rng_vec = []
    for rng, mn, count, i_start, i_end in rainflow.extract_cycles(soc_h2): 
        if soc_h2[i_start] - soc_h2[i_end] > 0:
            rng_vec.append(rng)
    if len(rng_vec)>0:
        soc_h2_max = max(rng_vec)

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
