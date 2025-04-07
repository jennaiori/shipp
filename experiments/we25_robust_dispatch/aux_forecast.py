
'''
Auxiliary functions for the numerical experiments related to robust dispatch optimization.

get_p_min_vec: generate a baseload constraint vector for a given reliability
mix_signals: mix two signals by linearly interpolating between them over a specified horizon
get_wf_power: calculate the power output of a wind farm based on wind speed magnitude and direction
'''

import numpy as np


def get_p_min_vec(p_min: float, data: np.ndarray, percent_min: float = 0.99,
                  return_len: bool = False) -> np.ndarray:
    '''
    Generate a baseload constraint vector for a given reliability

    The vector is constructed through an iterative process, where a
    variable len_continue_operation is increased progressively until
    the desired reliability level is reached. This variable refers
    to the maximum duration where the storage needs to cover the
    baseload constraint.

    Params:
        p_min (float): baseload power level [MW]
        data (np.array): power time series for which the baseload
            constraint need to be calculated [MW]
        percent_min (float, between 0 and 1): required reliability
            level
        return_len (bool): Boolean describing if the function needs
            to return the maximum duration during which the storage
            needs to cover the baseload constraint

    Returns:
        if return_len == True:
            len_continue_operation (int): maximum duration where the
                storage needs to cover the baseload constraint (in
                number of time steps)
        if return_len == False:
            p_min_vec (np.array): baseload constraint vector [MW]
    '''

    len_continue_operation = 0
    len_max = 240

    vec_99_pc = np.zeros_like(data)
    percent = sum(vec_99_pc)/len(vec_99_pc)

    m = len(data)
    while percent<percent_min and len_continue_operation<len_max:
        vec_99_pc = np.zeros_like(data)
        for i in range(m):
            if data[i] > p_min:
                vec_99_pc[i] = 1
            else:
                if i >= len_continue_operation:
                    value = 0
                    for j in range(len_continue_operation):
                        if data[i-(j+1)] > p_min:
                            value = 1
                    vec_99_pc[i] = value
        percent = sum(vec_99_pc)/len(vec_99_pc)
        len_continue_operation+=1

    assert (sum(vec_99_pc)/len(vec_99_pc))>= percent_min
    if len_continue_operation>= len_max-1:
        print('Warning get_p_min_vec: maximum length reached')

    if return_len:
        return len_continue_operation-1

    return p_min * vec_99_pc


def mix_signals(data1: list, data2: list, h: int) -> list:
    '''
    Mixes two signals by linearly interpolating between them over a specified horizon.
    
    Params:
        data1 (list): The first signal, a list of numerical values.
        data2 (list): The second signal, a list of numerical values.
        h (int): The horizon over which to mix the signals. This should be the length of data1 and data2.
    
    Returns:
        data_new: A new list containing the mixed signal values.
    '''

    data_new = []
    if h == 1:
        data_new.append(data1[0])
    else:
        for i in range(h):
            data_new.append(data1[i]*i/(h-1) + data2[i]*(1-i/(h-1))) 

    return data_new


def get_wf_power(wsp_mag : np.ndarray, wsp_dir : np.ndarray, parameters: dict):
    '''
    Calculate the power output of a wind farm based on wind speed magnitude and direction.
    
    Params:
        wsp_mag (float or np.ndarray): Wind speed magnitude in m/s.
        wsp_dir (float or np.ndarray): Wind direction in degrees.
        parameters (dict): object containing the parameters for the model
        
    Returns:
        np.ndarray: Power output of the wind farm in MW.
    '''
    # Reference values
    # offset = 1.1
    # factor = 0.35
    # period = 60
    # p_max = 10
    # n_turbines = 10
    # radius = 90
    # cp = 0.45
    # rho = 1.225
    # v_in = 4
    # v_out = 25

    offset = parameters["wake loss offset"]
    factor = parameters["wake loss factor"]
    period = parameters["wake loss period"]
    p_max = parameters["wind turbine rated power"]
    n_turbines = parameters["number of turbines"]
    radius = parameters["rotor radius"]
    cp = parameters["power coefficient"]
    rho = parameters["air density"]
    v_in = parameters["cut in wind speed"]
    v_out = parameters["cut out wind speed"]

    A = (radius**2)*np.pi
    p_th= np.array(0.5*cp*rho*A*(wsp_mag**3)*1e-6)

    omega = 2*np.pi*(1/period)
    wake_loss = np.minimum(1, offset -factor*np.cos(wsp_dir*omega) )

    p_th[wsp_mag<v_in or wsp_mag>v_out] = 0 

    return n_turbines*np.minimum(p_th*wake_loss, p_max)
