import json

import textwrap
import matplotlib.pyplot as plt
import numpy as np

# load data
folder = 'results/res_hkn/'

with open(folder+'data_res.json') as f:
    data_res = json.load(f)

labels = data_res['labels']
res_all = data_res['res_all']
nt = data_res['nt']

file_input = data_res['file_input']
with open( file_input) as f:
    data_input = json.load(f)
p_max = data_input['p_max']
p_min = data_input['p_min']
dt = data_input['dt']
n = data_input['n']              
rel = data_input['rel']

# Load forecast data!
print('Loading forecast data.')
file_windpower_vec = data_input['file_windpower']
forecast_vec = []
for file_windpower in file_windpower_vec:

    with open(file_windpower, 'r') as f:
        data_windpower = json.load(f)
    forecast_vec.append(data_windpower['windpower forecast'][:nt])

power = np.array(data_windpower['windpower observations'][:nt])


rel_og = sum([ 1/nt if p>=p_min else 0 for p in power[:nt]])
print('Wind power reliability: {:.2f}%'.format(rel_og*100), flush=True)
print('Target reliability: {:.2f}%'.format(rel*100), flush=True)

### Output values in terms of revenues
str_strategy = 'Strategy'
print('{}\tReliability\tRevenue [EUR]'.format(str_strategy.ljust(32)))

for res, label in zip(res_all, labels):
    print('{}\t{:6.1f}%\t\t{:7.0f}'.format(label.ljust(32), res['reliability']*100, res['revenues']))


# First figure
col_cy = "#00A6D6"
col_db = "#0C2340"
col_dg = "#00B8C8"
col_bl = "#0076C2"
col_pp = "#6F1D77"
col_pk = "#EF60A3"
col_dr = "#A50034"
col_rd = "#E03C31"
col_or = "#EC6842"
col_ye = "#FFB81C"
col_lg = "#6CC24A"
col_dg = "#009B77"

SMALL_SIZE = 7
MEDIUM_SIZE = 8
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rcParams['lines.markersize'] = 5
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1


col_all = [col_db, col_ye, col_lg, col_cy,  col_bl, col_db, col_db]

labels_new = ['Rule-based', 'Perfect Information', 
              'Real Information', r'Pessimistic $\alpha$ = 1', r'Pessimistic $\alpha$ = 2']


time_vec = np.arange(0, nt+n)/24

fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (7,1.7))

fig.subplots_adjust(wspace=0.3)
ax0.plot(time_vec[:nt], np.array(power[:nt]), 'k:', label = 'Wind power')


for (res, col) in zip(res_all, col_all):
    mrk = '-'
    if res['reliability']<rel-0.001:
        mrk = '-'
    ax0.plot(time_vec[:nt], np.array(power[:nt]) + np.array(res['power'][:nt]), mrk, color = col)


for (res, col) in zip(res_all, col_all):
    mrk = '-'
    if res['reliability']<rel-0.001:
        mrk = '-'
    ax1.plot(time_vec[:nt], res['energy'][:nt], mrk, color = col)


ax0.set_ylim([-0.1*p_max, 1.1*p_max])
ax0.set_xlabel('Time [days]')
ax0.set_ylabel('Power to the grid [MW]')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Energy [MWh]')

ax0.set_xlim([140, 147])
ax1.set_xlim([140, 147])

labels_new2 = ['Wind power']
for l in labels_new:
    labels_new2.append(l)

ax0.legend(labels_new2, loc = 'lower left', bbox_to_anchor = (0.35,1), ncols = 3)



# Second figure
labels_new = ['Rule-based', 'PI-U', 'PI', 'RI', r'P $\alpha$=1', r'P $\alpha$=2']

labels_wrapped = [textwrap.fill(txt, 13) for txt in labels_new]

fig, ax2 = plt.subplots(1, 1, figsize = (3, 2))

cnt = 0
for (res, col) in zip(res_all, col_all):   
    ax2.bar([cnt], [res['revenues']*1e-3], alpha = 0.5, color = col)
    cnt+=1


ax2.set_xticks(range(len(res_all)))
ax2.set_xticklabels(labels_wrapped)
ax2.tick_params(axis = 'x', labelsize = SMALL_SIZE)
ax2.set_ylabel('Storage revenues [kEur]')

ax2bis = ax2.twinx()
ax2bis.plot([-0.5, len(res_all)+0.5], [100*rel, 100*rel], 'k:')
for i in range(0, len(res_all)):
    ax2bis.plot(i, res_all[i]['reliability']*100, 'o', color = col_all[i])
    if (rel - res_all[i]['reliability'])> 0.001:
        ax2bis.annotate('{:.1f}%'.format(res_all[i]['reliability']*100), xy = [i-0.3, res_all[i]['reliability']*100-0.4], fontsize = SMALL_SIZE)
    if (rel - res_all[i]['reliability'])< 0.001:
        ax2bis.annotate('{:.1f}%'.format(res_all[i]['reliability']*100), xy = [i-0.3, res_all[i]['reliability']*100+0.3], fontsize = SMALL_SIZE)

ax2bis.set_ylabel(r'Reliability [\%]')
ax2bis.set_ylim([rel*100-3.5, rel*100+1])
ax2bis.set_xlim([-0.5, (len(res_all)-1)+0.5])

# Third figure
day_og = 3
fig, ax = plt.subplots(1,1,figsize = (2.3*1.2, 1.2*1.2))

ax.plot(time_vec[:24*day_og+1], power[:24*day_og+1], 'k-')
ax.plot(time_vec[24*day_og:(24*day_og+n)], power[24*day_og:(24*day_og+n)],  '--', color ='k',label = labels_new[2])
ax.set_xlabel('Time [days]')
ax.set_ylabel('Power [MW]')

ax.plot(time_vec[24*day_og:(24*day_og+n)], forecast_vec[0][24*day_og][0][:n], '--', color = col_all[2], label = labels_new[3])

ax.plot(time_vec[24*day_og:(24*day_og+n)], forecast_vec[1][24*day_og][0][:n], '--',color = col_all[3], label = labels_new[4])

ax.plot(time_vec[24*day_og:(24*day_og+n)], forecast_vec[2][24*day_og][0][:n], '--',color = col_all[4], label = labels_new[5])

ax.legend(loc = 'lower left', bbox_to_anchor = (0.15, 1), ncol=2)
ax.set_xticks([2,3,4,5])
ax.set_xlim([2, day_og+2.1])
ax.set_ylim([-5, 105])

plt.show(block = True)

