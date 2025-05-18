'''
    This file provides an example for the use of the shipp package.
    In this example, the power and price time series are obtained from 
    renewable.ninja and ENTSO-E, respectively. The script assesses the 
    impact of limited forecast information for a ramp-limitation 
    dispatchability requirement.
'''

import numpy as np
import matplotlib.pyplot as plt


from shipp.kernel_pyomo import solve_lp_pyomo, run_storage_operation
from shipp.components import Storage, Production, TimeSeries
from shipp.kernel import solve_lp_sparse
from shipp.io_functions import get_power_price_data


eur_to_usd = 1.18

# Global input data for the numerical experiments
n = 17 * 24  # number of time steps
dt = 1 # time step duration [hour]
discount_rate = 0.03 #discount rate
n_year = 20  # Project duration [years]
p_min = 25
p_cost_res = 3000  # USD/MW

# Input data for the storage characteristics
p_cost = 150*1e3  # cost per power capacity for STS [USD/MW]
e_cost = 175 * 1e3 # cost per energy capacity for STS [USD/MWh]
eta = 0.85 #Round trip efficiency for STS

pyo_solver = 'mosek'

# Input data for the power and price time series
date_start = '2019-01-01'
date_end = '2019-02-02'
latitude = 52.52
longitude = 13.405
p_max = 100
dp_min = -6 

# # Input files for the tokens
# filename_token_rninja = 'token_rninja.txt'
# filename_token_entsoe = 'token_entsoe.txt'

# # Retrieve the power and price data
# with open(filename_token_rninja, 'r') as file:
#     token_rninja = file.read().replace('\n', '')
# with open(filename_token_entsoe, 'r') as file:
#     token_entsoe = file.read().replace('\n', '')


# data_power, data_price = get_power_price_data(token_rninja, token_entsoe, date_start, date_end,    
#                                               latitude, longitude, capacity = 100*1e3)

import pandas as pd
file_data = "C:/Users/jiori/Codes/2 - HPP operation/data/input_files/simple_data_hkn.csv"
data = pd.read_csv(file_data)
data_price = data['price [EUR/MWh]'].values
data_power = data['Power [MW]'].values

k = 1 # parameter describing how the power ramp is calculated
dpower = data_power[k:n] - data_power[:n-k]

# Building storage and production objects
stor = Storage(e_cap = 126, p_cap = 27, eff_in = 1, eff_out= eta, 
               e_cost = e_cost, p_cost = p_cost)
stor_null = Storage(e_cap = 0, p_cap = 0, eff_in =1, eff_out=1, 
                    e_cost = 0, p_cost = 0)

price_dam = TimeSeries(data_price[:n]*eur_to_usd, dt)

power_ts = TimeSeries(data_power[:n], dt)
prod = Production(power_ts, p_cost_res)
prod_null = Production(TimeSeries([0 for _ in range(n)], dt), 0)

#calculate yearly revenues and reliability for renewable power only
revenues_res_only = np.dot(data_price[:n], np.minimum(data_power[:n], p_max))*dt
rel_ramp_res_only = sum([1/(n-1) if dp>=dp_min else 0 for dp in dpower ])
kmax =  np.argmin(dpower[:n-k]) # Index of minimum ramp

# Solve the sizing optimization problem
os =  solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, 0, p_max, n, pyo_solver, fixed_cap = True, dp_min = dp_min)

power_piu = np.array(os.power_out.data)
dpower_piu = power_piu[k:n] - power_piu[:n-k]

revenues_piu = np.dot(data_price[:n], np.minimum(power_piu[:n], p_max))*dt



print('Optimal storage size (MW/MWh):\t{:.2f}\t{:.2f}'.format(os.storage_list[0].p_cap, os.storage_list[0].e_cap))

print('\tdp_min [MW]\tRevenue [kUSD]\tRev. increase\tReliability\tMin. ramp [MW]')
print('RES\t{:.1f}\t\t{:.1f}\t\t{:.2f}%\t\t{:.1f}%\t\t{:.2f}'.format(0, revenues_res_only*1e-3, 0, rel_ramp_res_only*100, dpower[kmax]))
print('PI-U\t{:.1f}\t\t{:.1f}\t\t{:.2f}%\t\t{:.1f}%\t\t{:.2f}'.format(dp_min, revenues_piu*1e-3, 100*(revenues_piu/revenues_res_only-1), 100, dpower_piu[kmax]))


# Evaluate performance for limited forecast information
n_for = 120
rel_th = 1.0
n_hist = 120
forecast_perfect = [ [[p for p in data_power[init_index:init_index+n_for]]]  for init_index in range(0, n)]
e_start = os.storage_e[0].data[0]



# forecast_perfect = [ [[data_power[i+j] for j in range(n_for)]]  for i in range(n)]
res = run_storage_operation('forecast', data_power, data_price, 0, p_max, os.storage_list[0], e_start, n_for, n, dt, rel_th,  forecast_perfect, name_solver = 'mosek', dp_min = dp_min, verbose = False)


power_pi = np.array([data_power[i] + res['power'][i] for i in range(len(res['power']))])
dpower_pi = power_pi[k:n] - power_pi[:n-k]
revenues_pi = np.dot(data_price[:n], np.minimum(power_pi[:n], p_max))*dt



print('PI\t{:.1f}\t\t{:.1f}\t\t{:.2f}%\t\t{:.2f}%\t\t{:.2f}'.format(dp_min, revenues_pi*1e-3, 100*(revenues_pi/revenues_res_only-1), res['reliability']*100, dpower_pi[kmax]))

fig, ax = plt.subplots(1, 4, figsize = (15, 5))

ax[0].plot(dpower[:n], 'k--', label = 'Wind only')
ax[0].plot(dpower_piu, label = 'PI-U')
ax[0].plot(dpower_pi, label = 'PI')
ax[0].plot([kmax, kmax], [-15, 15], 'k:')
ax[0].legend()
ax[0].set_xlim([kmax-20, kmax+20])
ax[0].set_ylim([min(dpower), max(dpower)])
ax[0].set_xlabel('Time step [-]')
ax[0].set_ylabel('Power ramp [MW]')

ax[1].hist(dpower[:n], bins = 20, color = 'k')
ax[1].hist(dpower_piu, bins = 20, color = 'tab:blue')
ax[1].hist(dpower_pi, bins = 20, color = 'tab:orange')

ax[2].plot(data_power[:n], 'k--')
ax[2].plot(power_piu[:n])
ax[2].plot(power_pi[:n])
ax[2].plot([kmax, kmax], [0, p_max], 'k:')
ax[2].set_xlim([kmax-20, kmax+20])
ax[2].set_xlabel('Time step [-]')
ax[2].set_ylabel('Power [MW]')


ax[3].plot(os.storage_e[0].data)
ax[3].plot(res['energy'])
ax[3].set_xlim([kmax-20, kmax+20])
ax[2].plot([kmax, kmax], [0, os.storage_list[0].e_cap+1], 'k:')
ax[3].set_ylim([0, os.storage_list[0].e_cap+1])
ax[3].set_xlabel('Time step [-]')
ax[3].set_ylabel('Storage SoC [MWh]')
plt.show()