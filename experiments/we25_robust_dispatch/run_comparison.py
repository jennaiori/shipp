"""
This script performs a comparison of different storage operation strategies 
for wind power dispatch using various forecast methods and reliability targets.

Arguments:
    file_input (str): Path to the input JSON file containing dispatch parameters. Default: 'dispatch_input_file_hkn.json'.
    folder_out (str): Path to the output folder for saving results. Must end with a "/" and be a valid directory.
    nt (int, optional): Number of time steps for the simulation. Overrides the value in the input file. Default is 8600.

Usage:
    python run_comparison.py <file_input> <folder_out> [nt]

Output:
    Prints the reliability and revenue results for each dispatch strategy tested.
    Saves the results to a JSON file in the specified output folder.

"""

import json
import sys
import os
import time

from shipp.kernel_pyomo import run_storage_operation
from shipp.components import Storage

import numpy as np

name_solver = 'mosek'

file_input = 'dispatch_input_file_hkn.json'
folder_out = ''

if len(sys.argv) > 1:
    file_input = sys.argv[1]

if len(sys.argv) > 2:
    folder_out = sys.argv[2]
    assert folder_out[-1] == "/"
    assert os.path.isdir(folder_out)

print('Input file: ', file_input)

with open(file_input) as f:
    data_input = json.load(f)

# Load input parameters
p_max = data_input['p_max']
p_min = data_input['p_min']
dt = data_input['dt']
nt = data_input['nt']
n = data_input['n']              
eta1 = data_input['eta']
p_cap1 = data_input['p_cap']
e_cap1 = data_input['e_cap']
e_start = data_input['e_start']
rel_th = data_input['rel']
n_hist = data_input['n_hist']

if len(sys.argv) > 3:
    nt = int(sys.argv[3])

print('p_max:\t',p_max)
print('p_min:\t',p_min)
print('p_cap1:\t',p_cap1)
print('e_cap1:\t',e_cap1)
print('e_start:\t',e_start)
print('rel_th\t',rel_th)
print('n_hist\t',n_hist)
print('nt:\t', nt)

# Load price data
with open(data_input['file_price']) as f:
    data_price = json.load(f)

price = np.array(data_price['price [EUR/MWh]'])

# Load forecast data
file_windpower_vec = data_input['file_windpower']
forecast_vec = []
for file_windpower in file_windpower_vec:

    with open(file_windpower, 'r') as f:
        data_windpower = json.load(f)
    forecast_vec.append(data_windpower['windpower forecast'])


windpower_obs = np.array(data_windpower['windpower observations'])


# Check validity of input data
nt_max = min(len(price), len(windpower_obs))
print(n, nt, nt_max)
print(len(price), len(windpower_obs))
assert n+nt <= nt_max

assert(min(price)>0)

# Generate perfect information forecast
# forecast_perfect = get_forecast_set('perfect', windpower_obs, p_max, n, nt, {})
forecast_perfect = [ [[p for p in windpower_obs[init_index:init_index+n]]]  for init_index in range(0, nt)]


# Check wind reliability
rel_og = sum([ 1/nt if p>=p_min else 0 for p in windpower_obs[:nt]])
print('Wind power reliability: {:.2f}%'.format(rel_og*100), flush=True)
print('Target reliability: {:.2f}%'.format(rel_th*100), flush=True)

# Generate storage object for the rule-based strategy
stor = Storage(e_cap = e_cap1,  p_cap = p_cap1, eff_in = 1.0, eff_out = eta1)

# Run operation cases
labels = ['Rule-based', 'Perfect Information Unlimited', 'Perfect Information Limited',
           'Real Information',  'Pessimistic A', 'Pessimistic B']

print('Run operation cases:', flush=True)

res_all = []
cnt = 0

start_t = time.time()
res_rub = run_storage_operation('rule-based', windpower_obs, price, p_min, p_max, stor, e_start, n, nt, dt, rel = rel_th)
print('\t', labels[cnt], '\t(time: {:6.1f}m)\t rel: {:.2f}% \tRevenues: {:.0f}EUR'.format((time.time()-start_t)/60, 100*res_rub['reliability'], res_rub['revenues']), flush=True)
res_all.append(res_rub)
cnt+=1

start_t = time.time()
res_det = run_storage_operation('unlimited', windpower_obs, price, p_min,
                                 p_max, stor, e_start, n, nt, dt, rel = rel_th, name_solver= name_solver)
print('\t', labels[cnt], '\t(time: {:6.1f}m)\t rel: {:.2f}% \tRevenues: {:.0f}EUR'.format((time.time()-start_t)/60, 100*res_det['reliability'], res_det['revenues']), flush=True)
res_all.append(res_det)
cnt+=1


start_t = time.time()
res_pi = run_storage_operation('forecast', windpower_obs, price, p_min, p_max, stor, e_start, n, nt, dt, rel = rel_th, n_hist = n_hist, forecast = forecast_perfect, name_solver= name_solver)
print('\t', labels[cnt], '\t(time: {:6.1f}m)\t rel: {:.2f}% \tRevenues: {:.0f}EUR'.format((time.time()-start_t)/60, 100*res_pi['reliability'], res_pi['revenues']), flush=True)
res_all.append(res_pi)
cnt+=1

for forecast_tmp in forecast_vec:
    start_t = time.time()
    res_for = run_storage_operation('forecast', windpower_obs, price, p_min, p_max, stor, e_start, n, nt, dt, rel = rel_th, n_hist = n_hist, forecast = forecast_tmp, name_solver= name_solver)
    print('\t', labels[cnt], '\t(time: {:6.1f}m)\t rel: {:.2f}% \tRevenues: {:.0f}EUR'.format((time.time()-start_t)/60, 100*res_for['reliability'], res_for['revenues']), flush=True)
    res_all.append(res_for)
    cnt+=1

print('Datafiles created:', flush=True)

with open(folder_out+'data_res.json', 'w', encoding = 'utf-8') as f:
    json.dump({'labels': labels, 'res_all': res_all, 'file_input': file_input, 'folder_out': folder_out, 
               'nt':nt}, f, ensure_ascii=False, indent = 4)
print('\t', folder_out+'data_res.json')


print('\nEnd', flush=True)
