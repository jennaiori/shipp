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
from shipp.kernel_pyomo import solve_dispatch_pyomo
from shipp.components import Storage, Production, TimeSeries
from shipp.kernel import solve_lp_sparse
from shipp.io_functions import get_power_price_data


eur_to_usd = 1.18

# Global input data for the numerical experiments
n = 5*24  # number of time steps
dt = 1 # time step duration [hour]
discount_rate = 0.03 #discount rate
n_year = 20  # Project duration [years]
p_min = 0
p_cost_res = 3000  # USD/MW
dp_lim = 6 

# Input data for the storage characteristics
p_cost = 150*1e3/eur_to_usd  # cost per power capacity for STS [USD/MW]
e_cost = 75 * 1e3/eur_to_usd # cost per energy capacity for STS [USD/MWh]
eta = 0.85 #Round trip efficiency for STS

pyo_solver = 'mosek'

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

k = 1 # parameter describing how the power ramp is calculated
dpower = data_power[k:] - data_power[:-k]

# Building storage and production objects
stor = Storage(e_cap = 20, p_cap = 10, eff_in = 1, eff_out= eta, 
               e_cost = e_cost, p_cost = p_cost)
stor_null = Storage(e_cap = 0, p_cap = 0, eff_in =1, eff_out=1, 
                    e_cost = 0, p_cost = 0)

price_dam = TimeSeries(data_price[:n], dt)

power_ts = TimeSeries(data_power[:n], dt)
prod = Production(power_ts, p_cost_res)
prod_null = Production(TimeSeries([0 for _ in range(n)], dt), 0)

#calculate yearly revenues and reliability for renewable power only
revenues_res_only = np.dot(data_price[:n], np.minimum(data_power[:n], p_max))*dt
rel_ramp_res_only = sum([1/(n-1) if( dp_lim >= dp>=-dp_lim )else 0 for dp in dpower[:n-1] ])
kmin =  np.argmin(dpower[:n-k]) # Index of minimum ramp
kmax =  np.argmax(dpower[:n-k]) # Index of maximum ramp

# Solve the sizing optimization problem
os =  solve_lp_pyomo(price_dam, prod, prod_null, stor, stor_null, discount_rate, n_year, 0, p_max, n, pyo_solver, fixed_cap = False, dp_lim = dp_lim, verbose = False)

power_piu = np.array(os.power_out.data)
dpower_piu = power_piu[1:] - power_piu[:-1]
revenues_piu = np.dot(data_price[:n], np.minimum(power_piu[:n], p_max))*dt


print('Optimal storage size (MW/MWh):\t{:.2f}\t{:.2f}'.format(os.storage_list[0].p_cap, os.storage_list[0].e_cap))

print('\tdp_lim [MW]\tRevenue [kUSD]\tReliability\tMin. ramp [MW/h]\t Max. ramp [MW/h]')
print('RES\t{:.1f}\t\t{:.1f} ({:.1f}%)\t{:.1f}%\t\t{:.1f}\t\t\t{:.1f}'.format(0, revenues_res_only*1e-3, 0, rel_ramp_res_only*100, dpower[kmin], dpower[kmax]))
print('PI-U\t{:.1f}\t\t{:.1f} ({:.1f}%)\t{:.1f}%\t\t{:.1f}\t{:.1f}\t\t{:.1f}\t{:.1f}'.format(dp_lim, revenues_piu*1e-3, 100*(revenues_piu/revenues_res_only-1), 100, dpower_piu[kmin], min(dpower_piu), dpower_piu[kmax], max(dpower_piu)))

# Evaluate performance for limited forecast information
n_for = 12
rel_th = 1.0
n_hist = 0
forecast_perfect = [ [[p for p in data_power[init_index:init_index+n_for]]]  for init_index in range(0, n)]
e_start = os.storage_e[0].data[0]
mu = 1e4
beta_obj = 1e-6

res = run_storage_operation('forecast', data_power, data_price, 0, p_max, os.storage_list[0], e_start, n_for, n, dt, rel_th,  forecast_perfect, name_solver = 'mosek', dp_lim = dp_lim, verbose = False, mu = mu, beta_obj = beta_obj)

power_pi = np.array([data_power[i] + res['power'][i] - res['p_cur'][i] for i in range(len(res['power']))])
dpower_pi = power_pi[k:] - power_pi[:-k]
revenues_pi = np.dot(data_price[:n], np.minimum(power_pi[:n], p_max))*dt


print('PI\t{:.1f}\t\t{:.1f} ({:.1f}%)\t{:.1f}%\t\t{:.1f}\t{:.1f}\t\t{:.1f}\t{:.1f}'.format(dp_lim, revenues_pi*1e-3, 100*(revenues_pi/revenues_res_only-1), res['reliability']*100, dpower_pi[kmin], min(dpower_pi), dpower_pi[kmax], max(dpower_pi)))

fig, ax = plt.subplots(1, 4, figsize = (15, 5))
fig.subplots_adjust(wspace = 0.3)
ax[0].plot(dpower[:n], 'k--', label = 'Wind only')
ax[0].plot(dpower_piu, label = 'PI-U')
ax[0].plot(dpower_pi, label = 'PI')
ax[0].legend()
ax[0].set_ylim([min(dpower), max(dpower)])
ax[0].set_xlabel('Time step [-]')
ax[0].set_ylabel('Power ramp [MW]')

ax[1].plot(data_power[:n], 'k--')
ax[1].plot(power_piu[:n])
ax[1].plot(power_pi[:n])
ax[1].set_xlabel('Time step [-]')
ax[1].set_ylabel('Power [MW]')

ax[2].plot(os.storage_e[0].data)
ax[2].plot(res['energy'])
ax[2].set_ylim([0, os.storage_list[0].e_cap+1])
ax[2].set_xlabel('Time step [-]')
ax[2].set_ylabel('Storage SoC [MWh]')

bin_hist = np.linspace( dpower[kmin],  dpower[kmax], 20)

ax[3].hist(dpower[:n], bins = bin_hist, color = 'k')
ax[3].hist(dpower_piu, bins = bin_hist, color = 'tab:blue')
ax[3].hist(dpower_pi, bins = bin_hist, color = 'tab:orange', alpha = 0.5)
ax[3].set_xlabel('Power ramp [MW]')
ax[3].set_ylabel('Count [-]')

ax[1].set_xlim([kmin-20, kmin+20])
ax[2].set_xlim([kmin-20, kmin+20])
ax[0].set_xlim([kmin-20, kmin+20])
ax[0].plot([kmin, kmin], [-15, 15], 'k:')
ax[1].plot([kmin, kmin], [0, p_max], 'k:')
ax[1].plot([kmin, kmin], [0, os.storage_list[0].e_cap+1], 'k:')

# Analyse the first 3 time steps of the dispatch optimization
verbose = False
nt = 3

name_solver = 'mosek'
p_res = []
p_cur_res = []
e_res = [e_start]
bin_res = []
e_start_new = e_start

m = len(forecast_perfect[0]) # Number of forecast_perfect scenarios.

p_hist_res   =  data_power[0]
p_hist_stor   =  0
cnt_hist = n_hist # Initializing the count of historical time steps meeting the power threshold.

fig, ax = plt.subplots(4, nt, figsize = (2*nt, 8))
for t in range(nt):
    ax[0, t].plot( os.storage_p[0].data[:(nt+n_for)] )
    ax[1, t].plot( os.power_out.data[:(nt+n_for)])
    ax[2, t].plot( os.storage_e[0].data[:(nt+n_for)] )
    ax[3, t].plot( data_power[:nt+n_for] - os.production_p[0].data[:nt+n_for] )

# Iterate over the time steps in the simulation for the rolling horizon.
for t in range(nt):
    # If the current time step is lower than the length of the time window for past operation, we use a smaller time window for the optimization.
    if t > n_hist:
        cnt_hist = sum([0 if p+ps < p_min else 1 for ps, p in zip(p_res[-n_hist:], data_power[t-n_hist:t])])

        p_vec, e_vec, p_vec2, _, p_cur, bin_vec, status = solve_dispatch_pyomo(data_price[t:], m, rel_th, n_for, forecast_perfect[t], p_min, p_max, e_start_new, 0,  dt, stor, stor_null, n_hist = n_hist, cnt_hist=cnt_hist, verbose = verbose, name_solver = name_solver, dp_lim = dp_lim, beta_obj = beta_obj, mu = mu, p_hist_res = p_hist_res, p_hist_stor=p_hist_stor)
    else: 
        cnt_hist = sum([0 if p+ps < p_min else 1 for ps, p in zip(p_res[:t], data_power[:t])])

        p_vec, e_vec, p_vec2, _, p_cur, bin_vec, status = solve_dispatch_pyomo(data_price[t:], m, rel_th, n_for, forecast_perfect[t], p_min, p_max, e_start_new, 0,  dt, stor, stor_null, n_hist = n_hist, cnt_hist=(t-1), verbose = verbose, name_solver = name_solver, dp_lim = dp_lim, beta_obj = beta_obj, mu = mu, p_hist_res = p_hist_res, p_hist_stor=p_hist_stor)
    
    # If the optimization problem is solved correctly, we retrieve the results.
    if status == 'ok':
        e_start_new = e_vec[0,1]
        p_new = p_vec[0,0]
    else:
        print('Time step warning:', t, t/24, e_start_new)
        # If the optimization solver fails, calculate the power and energy levels so that the storage system charges or discharge to meet the baseload power level.
        delta_power = p_min - data_power[t]
        if delta_power >= 0:
            p_new = min(e_res[t]*stor.eff_out/dt, delta_power, stor.p_cap)
            e_start_new = e_res[t] - p_new*dt/stor.eff_out
        else:
            p_new = max(-1.0/dt*(stor.e_cap-e_res[t]), delta_power, -stor.p_cap)
            e_start_new = e_res[t] - p_new*dt

    p_hist_stor = p_new
    p_hist_res = data_power[t]-p_cur[0,0]
    if (p_hist_res + p_hist_stor) < -1e-4:
        print('p_hist <0 at t=',t,  'p_new=', p_new, 'data_power[t]=', data_power[t])
    e_res.append(e_start_new)
    p_res.append(p_new)
    bin_res.append(bin_vec[0])
    p_cur_res.append(p_cur[0,0])

    ax[0, t].plot([x for x in range(t, t+n_for)], p_vec[0,:] , '--')
    ax[1, t].plot([x for x in range(t, t+n_for)], p_vec[0,:] + data_power[t:t+n_for] - p_cur[0,:] , '--')
    ax[2, t].plot([x for x in range(t, t+n_for)], e_vec[0,:n_for] , '--')
    ax[3, t].plot([x for x in range(t, t+n_for)], p_cur[0,:], '--' )

rel_res = 1/nt*sum(bin_res)
rev_res = sum([data_price[i]*(p_res[i]-p_cur_res[i]) for i in range(nt)])
cost_res = -rev_res/(sum(data_power[:nt])*dt)

res = {'power':p_res, 'energy': e_res, 'reliability': rel_res, 
        'revenues': rev_res, 'cost': cost_res, 'p_cur': p_cur_res,
        'bin': np.array(bin_res).astype(int).tolist()}
for t in range(nt):
    ax[0, t].plot( p_res[:(nt)], ':' )
    ax[1, t].plot( np.array(p_res[:(nt)]) + np.array(data_power[:(nt)]), ':' )
    ax[2, t].plot( e_res[:(nt)] , ':' )

ax[0, 0].set_ylabel("Storage power [MW]")
ax[1, 0].set_ylabel("Power to grid [MW]")
ax[2, 0].set_ylabel("State of charge [MWh]")
ax[3, 0].set_ylabel("Curtailed power [MW]")
for t in range(nt):
    ax[3, t].set_xlabel("Time step [-]")

ax[0,0].legend(["PI-U", "PI forecasted", "PI actual"], loc = "lower left", bbox_to_anchor = (0.5, 1.05), ncols = 3)

plt.show()
