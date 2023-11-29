'''
    Module used to define "kernel" functions, i.e. functions that are 
    used throughout the code for various calculations.

'''

import numpy as np
from scipy.signal import correlate
from scipy.optimize import linprog
import matplotlib.pyplot as plt

from sizing_opt_hpp.components import Storage, OpSchedule
from sizing_opt_hpp.timeseries import TimeSeries

def power_calc(wind, radius, cp, v_in, v_r, v_out, p_max = None):
    '''
        Function to calculate the power output of wind based on the wind
        data (wind), the rotor radius (radius), the power coefficient (cp),
        the cut-in, rated and cut-out wind speed (v_in, v_r, v_out)
        The function output the power time series (power) in MW
    '''
    rho = 1.225
    pi = 3.141591
    power = wind**3 * 0.5 * rho * pi * radius **2 * cp * 1e-6
    power[np.logical_or(wind<v_in,wind>v_out)] = 0
    if p_max is None:
        power[wind>= v_r] = v_r**3 * 0.5 * rho * pi * radius **2 * cp * 1e-6
    else:
        power[power>= p_max] = p_max
    return power

def plot_xcorr(x, y, delta_t = 1/24):
    '''
        Plot cross-correlation (full) between two signals.
    '''
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

def build_lp_cst(power, dt, p_min, p_max, n, losses_batt, losses_h2):
    '''
        Design variables:
            - Power from wind [n]
            - Power from battery (charge/discharge) [n]
            - Power from fuel cell (>0) or electrolyzer (<0 [n]
            - Losses from battery [n] (>0)
            - Losses from electrolyzer [n] (>0)
            - State of charge from battery [n+1]
            - Hydrogen levels [n+1]
            - Max state of charge (battery capacity) [1]
            - Max hydrogen power capacity (battery capacity) [1]
    '''
    init_batt_charge = 0
    init_h2_levels = 0
    # losses_batt = 0.10 # Round-trip efficiency from Mehta et. al 2019 is 90%
    # losses_h2 = 0.7
    # losses_fc = 0.4
    # losses_h2 = 0.5 * (losses_h2 - losses_fc)

    rate_batt = p_max
    rate_fc = p_max
    rate_h2 = p_max

    max_soc = n*dt*rate_batt #MWh
    max_h2 = 100*1400*0.0333 #MWh  (1400kg H2 and 1kg is 33.3 kWh)

    z_n = np.zeros((n,n))
    z_1n = np.zeros((1,n))
    z_n1 = np.zeros((n, 1))

    # EQUALITY CONSTRAINTS
    # Constraint on wind power + battery power + h2 power >= p_min
    matrix_power_bound = np.hstack((np.eye(n), np.eye(n), np.eye(n), z_n, z_n,
                                    np.zeros((n,n+1)), np.zeros((n,n+1)), z_n1,
                                    z_n1))
    vector_power_min = np.ones((n,1))* p_min
    vector_power_max = np.ones((n,1))* p_max

    # Constraint on state of charge:
    # soc_(n) - soc_(n+1) - dt * (P_batt - losses) = 0
    matrix_last_soc = np.vstack((np.zeros((n-1,1)), np.array([[-1]])))
    matrix_c2 = np.hstack((z_n, -dt * np.eye(n), z_n, dt * np.eye(n),  z_n,
                           np.eye(n) - np.diag(np.ones(n-1), 1),
                           matrix_last_soc,  np.zeros((n,n+1)), z_n1, z_n1))
    vector_b2 = z_n1
    # Constraint on first state of charge
    matrix_c3 = np.hstack((z_1n, z_1n, z_1n, z_1n, z_1n, np.array([[1]]),
                           np.zeros((1, n)), np.zeros((1,n+1)),
                           -init_batt_charge*np.ones((1,1)), np.zeros((1,1))))
    vector_b3 = np.zeros((1,1))

    # Constraint on first h2 level
    matrix_c5 = np.hstack((z_1n, z_1n, z_1n, z_1n, z_1n, np.zeros((1,n+1)),
                           np.array([[1]]), np.zeros((1, n)),
                           -init_h2_levels*np.ones((1,1)), np.zeros((1,1))))
    vector_b5 = np.zeros((1,1))

    # Constraint on the battery losses
    matrix_losses_batt = np.hstack((z_n, losses_batt*np.eye(n),z_n, -1*np.eye(n),
                                    np.zeros((n, n)), np.zeros((n, n+1)),
                                    np.zeros((n, n+1)), z_n1, z_n1))
    vector_losses_batt = z_n1
    # Constraint on the h2 losses
    matrix_losses_h2 = np.hstack((z_n,z_n, losses_h2*np.eye(n),
                                  np.zeros((n, n)), -1*np.eye(n),
                                  np.zeros((n, n+1)), np.zeros((n, n+1)), z_n1,
                                  z_n1))
    vector_losses_h2 = z_n1

    # INEQ CONSTRAINT
    # Constraint on the maximum state of charge (i.e battery capacity)
    matrix_max_soc = np.hstack((np.zeros((n+1,n)), np.zeros((n+1, n)),
                                np.zeros((n+1, n)), np.zeros((n+1,n)),
                                np.zeros((n+1,n)), np.eye(n+1),
                                np.zeros((n+1, n+1)), -1*np.ones((n+1,1)),
                                np.zeros((n+1, 1))))
    vector_max_soc = np.zeros((n+1,1))


    matrix_max_batt = np.hstack((z_n, np.eye(n), z_n, z_n, z_n,
                                 np.zeros((n,n+1)), np.zeros((n,n+1)),
                                 -1 * np.ones((n,1)), z_n1))
    vector_max_batt = z_n1

    # Constraint on hydrogen levels:  h2_(n) - h2_(n+1) - dt * (P_el - losses_h2 - losses_fc) >=0
    matrix_last_soc = np.vstack((np.zeros((n-1,1)), np.array([[1]])))
    matrix_h2_soc = np.hstack((z_n,  z_n, dt * np.eye(n),  z_n, -dt*np.eye(n),
                               np.zeros((n,n+1)),
                               -np.eye(n) + np.diag(np.ones(n-1), 1),
                               matrix_last_soc, z_n1, z_n1))
    vector_h2_soc = z_n1

    # Constraint on electrolyzer / fuel cell maximum power output
    matrix_h2_max_power = np.hstack((z_n, z_n, np.eye(n), z_n, z_n,
                                     np.zeros((n,n+1)), np.zeros((n,n+1)),
                                     z_n1, -np.ones((n,1))))
    vector_h2_max_power = z_n1
    matrix_h2_min_power = np.hstack((z_n, z_n, -np.eye(n), z_n, z_n,
                                     np.zeros((n,n+1)), np.zeros((n,n+1)),
                                     z_n1, -np.ones((n,1))))
    vector_h2_min_power = z_n1



    matrix_a = np.vstack((matrix_c2, matrix_c3,  matrix_c5, matrix_losses_batt,
                          matrix_losses_h2, matrix_h2_soc))
    vector_b = np.vstack((vector_b2, vector_b3,  vector_b5, vector_losses_batt,
                          vector_losses_h2, vector_h2_soc))
    # print(matrix_a)
    # print(vector_b)


    matrix_a_ineq = np.vstack((-1*matrix_power_bound, matrix_power_bound,
                               matrix_max_soc, matrix_h2_max_power,
                               matrix_h2_min_power, matrix_max_batt))
    vector_b_ineq = np.vstack((-1*vector_power_min, vector_power_max,
                               vector_max_soc, vector_h2_max_power,
                               vector_h2_min_power, vector_max_batt))
    # BOUNDS ON DESIGN VARIABLES
    bounds_lower = np.vstack((np.zeros((n,1)),
                              -rate_batt * np.ones((n,1)),
                              -rate_h2 * np.ones((n,1)),
                              -rate_h2*np.ones((n,1)),
                              -rate_h2*np.ones((n,1)),
                              np.zeros((n+1,1)),
                              np.zeros((n+1,1)),
                              np.zeros((1,1)),
                              np.zeros((1,1))))
    bounds_upper = np.vstack((power[0:n].reshape(n,1),
                              rate_batt * np.ones((n,1)),
                              rate_fc * np.ones((n,1)),
                              rate_h2*np.ones((n,1)),
                              rate_h2*np.ones((n,1)),
                              max_soc*np.ones((n+1,1)),
                              max_h2*np.ones((n+1,1)),
                              max_soc*np.ones((1,1)),
                              rate_h2 * np.ones((1,1))))


    bounds = []
    for x in range(0, len(bounds_lower)):
        bounds.append((bounds_lower[x][0], bounds_upper[x][0]))

    return matrix_a, vector_b, matrix_a_ineq, vector_b_ineq, bounds

def build_lp_obj(power, price, n, eta, alpha):
    '''
        Design variables:
            - Power from wind [n]
            - Power from battery (charge/discharge) [n]
            - Power from fuel cell (>0) or electrolyzer (<0 [n]
            - Losses from battery [n] (>0)
            - Losses from electrolyzer [n] (>0)
            - State of charge from battery [n+1]
            - Hydrogen levels [n+1]
            - Max state of charge (battery capacity) [1]
            - Max hydrogen power capacity (battery capacity) [1]
    '''
    normed_price = np.reshape(price[:n], (n,1))/(sum(price[:n])*np.median(power[:n]))
    vector_c = np.vstack((-eta*normed_price,             # Wind power
                      -eta*normed_price,             # Batt power
                      -eta*normed_price,             # Power from fuel cell / to electrolyzer
                      0.0*np.ones((n,1)),      # minimize losses from batteries
                      0.0*np.ones((n,1)),      #minimize losses from electrolizer
                      np.zeros((n+1, 1)),           # SoC
                      np.zeros((n+1, 1)),           # H2 levels
                      (1-eta)*alpha*np.ones((1,1)),           # minimize max state of charge
                      (1-eta)*(1-alpha)*np.ones((1,1))))          # minimize max h2 power rate

    return vector_c

def solve_lp(power_ts, price_ts, stor_batt: Storage, stor_h2: Storage,
             eta, alpha, p_min, p_max, n):
    '''
        Solves hybrid sizing problem as a linear program
    '''
    assert power_ts.dt == price_ts.dt

    dt = power_ts.dt

    losses_batt = 1 - stor_batt.eff_in * stor_batt.eff_out
    losses_h2 = 1 - stor_h2.eff_in * stor_h2.eff_out

    vec_obj = build_lp_obj(power_ts.data, price_ts.data, n, eta, alpha)
    mat_eq, vec_eq, mat_ineq, vec_ineq, bounds = build_lp_cst(power_ts.data,
                                                              dt,
                                                              p_min, p_max, n,
                                                              losses_batt,
                                                              losses_h2)

    res = linprog(vec_obj, A_ub= mat_ineq, b_ub = vec_ineq, A_eq=mat_eq,
                  b_eq=vec_eq, bounds=bounds, method = 'highs')

    print(res.message)
    power_wind = res.x[0:n]
    power_batt = res.x[n:2*n]
    power_h2 = res.x[2*n:3*n]
    # power_losses = res.x[3*n:4*n]
    # power_losses_h2 = res.x[4*n:5*n]
    soc = res.x[5*n:6*n]
    h2 = res.x[6*n+1:7*n+1]
    # final_h2 = res.x[7*n+1]
    batt_capacity = res.x[7*n+2]
    max_h2_power = res.x[7*n+3]

    print('Battery energy cap [MWh] = ', batt_capacity)
    print('Battery power cap [MW] = ', np.max(power_batt))
    print('H2 power cap [MW] = ', max_h2_power)
    print('H2 energy cap [MWh eq.] = ', np.max(h2))

    power_out = power_wind + power_batt + power_h2
    revenues = np.dot(price_ts.data[:n], power_out) * dt
    print("Revenues [kEur] =", revenues * 1e-3)

    print("Percent of wind only revenues [kEur] =",
          100* revenues / (np.dot(price_ts.data[:n], power_ts.data[:n])))

    stor_batt_res = Storage(e_cap = batt_capacity,
                            p_cap = max(power_batt),
                            eff_in = 0,
                            eff_out = 1-losses_batt)
    stor_h2_res = Storage(e_cap = np.max(h2),
                            p_cap = max_h2_power,
                            eff_in = 0,
                            eff_out = 1-losses_h2)

    os_res = OpSchedule(production_list = [None],
                        storage_list = [stor_batt_res, stor_h2_res],
                        production_p = [TimeSeries(power_wind, dt)],
                        storage_p = [TimeSeries(power_batt, dt),
                                     TimeSeries(power_h2, dt)],
                        storage_e = [TimeSeries(soc, dt),
                                     TimeSeries(h2, dt)])

    return os_res
