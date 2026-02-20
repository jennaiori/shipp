'''
    This file provides an example for the use of the shipp package.
    In this example, the power and price time series are obtained from renewable.ninja and ENTSO-E, respectively. 
    Two problems are solved: 
        - an integrated sizing and dispatch optimization problem
        - a dispatch optimization problem, where the size of the storage is fixed
'''

import numpy as np
import matplotlib.pyplot as plt


from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries
from shipp.kernel import solve_lp_sparse
from shipp.io_functions import get_power_price_data


# Global input data for the numerical experiments
n = 20 * 24  # number of time steps
dt = 1 # time step duration [hour]
discount_rate = 0.03 #discount rate
n_year = 20  # Project duration [years]
p_min = 10 # Required minimum baseload power
p_cost_res = 3000  # cost per installed capacity of the renewable energy source [USD/MW]

# Input data for the storage characteristics
p_cost = 150*1e3  # cost per power capacity for the short term storage [USD/MW]
e_cost = 75 * 1e3 # cost per energy capacity for the short term storage [USD/MWh]
eta = 0.85 #Round trip efficiency for the short term storage

pyo_solver = 'none' # Name of the optimization solver (e.g. cplex, gurobi, mosek) - if 'none', default to linprog

# Input data for the power and price time series
date_start = '2019-01-01'
date_end = '2019-02-02'
latitude = 52.52
longitude = 13.405
p_max = 100

# Input files for the tokens
filename_token_rninja = 'token_rninja.txt'
filename_token_entsoe = 'token_entsoe.txt'

# Retrieve the power and price data
with open(filename_token_rninja, 'r') as file:
    token_rninja = file.read().replace('\n', '')
with open(filename_token_entsoe, 'r') as file:
    token_entsoe = file.read().replace('\n', '')

data_power, data_price = get_power_price_data(token_rninja, token_entsoe, date_start, date_end, latitude, longitude, capacity = 100*1e3)

# Building objects representing the storage system
stor = Storage(e_cap = 500, p_cap = 50, eff_in = 1, eff_out= eta, e_cost = e_cost, p_cost = p_cost)
stor_null = Storage(e_cap = 0, p_cap = 0, eff_in =1, eff_out=1, e_cost = 0, p_cost = 0)  # Creation of a null object - necessary for the optimization function

price_dam = TimeSeries(data_price[:n], dt)

# Building objects representing the power production assets
power_ts = TimeSeries(data_power[:n], dt)
prod = Production(power_ts, p_cost_res)
prod_null = Production(TimeSeries([0 for _ in range(n)], dt), 0)  # Creation of a null object - necessary for the optimization function

# Solve the sizing optimization problem and compare with fixed sizing
if pyo_solver == 'none':
    if n>150*24:
        n = 150*24
        print('Number of time steps limited to 3600 due to the poor performance of solver linprog on large problems.')
    
    # Solve the integrated sizing + dispatch problem
    os =  solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n)
    # Solve the dispatch problem with fixed storage capacity
    os_fixed =  solve_lp_sparse(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, fixed_cap = True)

else:
    # Solve the integrated sizing + dispatch problem
    os =  solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, pyo_solver)
     # Solve the dispatch problem with fixed storage capacity
    os_fixed =  solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, pyo_solver, fixed_cap = True)

#calculate yearly revenues for renewable power only
revenues_res_only = np.dot(data_price[:n], np.minimum(data_power[:n], p_max))*dt

# Calculate the "Added" NPV, meaning the difference of NPV due to the addition of a storage system  
os.get_added_npv(discount_rate, n_year)
os_fixed.get_added_npv(discount_rate, n_year)

print('\t\t\tP_min [MW]\tRevenue [kUSD]\tRev. increase\tp_cap1/e_cap1\t\
      p_cap2/e_cap2\tCost BL [M.USD]\tTot NPV [M.USD]')
print('Dispatch + Sizing\t{:.1f}\t\t{:.1f}\t\t{:.2f}%\t\t{:.2f}/{:.2f}\t\t{:.2f}/{:.2f}\
      \t{:.2f}\t\t\t{:.1f}'.format(p_min, os.revenue*1e-3, 
                                 100*(os.revenue/revenues_res_only-1),
                                 os.storage_list[0].p_cap, 
                                 os.storage_list[0].e_cap, 
                                 os.storage_list[1].p_cap, 
                                 os.storage_list[1].e_cap, -os.a_npv, os.npv))
print('Dispatch only\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}%\t\t{:.2f}/{:.2f}\t\t{:.2f}/{:.2f}\
      \t{:.2f}\t\t\t{:.1f}'.format(p_min, os_fixed.revenue*1e-3, 
                                 100*(os_fixed.revenue/revenues_res_only-1),
                                 os_fixed.storage_list[0].p_cap, 
                                 os_fixed.storage_list[0].e_cap, 
                                 os_fixed.storage_list[1].p_cap, 
                                 os_fixed.storage_list[1].e_cap, -os_fixed.a_npv, os_fixed.npv))



# Plot the time series of power and state of charge
time_vec = np.arange(n)*1/24
fig, ax = plt.subplots(1,2, figsize = (10, 5))

ax[0].plot(time_vec, data_power[:n] + os.storage_p[0].data, label = 'Power + optimal storage')
ax[0].plot(time_vec, data_power[:n] + os_fixed.storage_p[0].data, label = 'Power + fixed storage')
ax[0].plot(time_vec, data_power[:n], label = 'Wind power')
ax[0].legend()
ax[0].set_xlabel('Time [days]')
ax[0].set_ylabel('Power [MW]')

ax[1].plot(time_vec, os.storage_e[0].data, label = 'Optimal storage')
ax[1].plot(time_vec, os_fixed.storage_e[0].data, label = 'Fixed storage')
ax[1].legend()
ax[1].set_xlabel('Time [days]')
ax[1].set_ylabel('State of charge [MWh]')

plt.show()