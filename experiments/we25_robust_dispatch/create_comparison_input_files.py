
"""
This script generates input files for a dispatch optimization problem based on 
wind power and price data. It calculates storage capacity and other parameters 
required for the optimization, and outputs the results in a JSON file.

Arguments:
    folder_in (str): Path to the input data folder. Must end with a '/'.
    folder_out (str): Path to the output data folder. Must end with a '/'.
    filename (str): Path to the CSV file containing site information.
    site (str): Name of the site for which the input files are generated.

Optional arguments:
    n (int): Number of time steps for the optimization horizon. Default is 48.
    nt (int): Total number of time steps in the dataset. Default is 8600.
    rel (float): Reliability level for minimum power calculation. Default is 0.99.
    p_min (float): Minimum power level for the wind power observations. Default is 10.
    n_hist (int): Number of historical time steps to consider. Default is 168.
    duration (float): Duration for which storage is sized (in hours). Default is 0 (auto-sizing).
    extra_power (float): Additional power capacity for storage. Default is 0.
    dt (float): Time step duration (in hours). Default is 1.

Usage:
    python create_comparison_input_files.py <folder_in> <folder_out> <filename> <site> [n] [nt] [rel] [p_min] [n_hist] [duration] [extra_power]

Output:
    A JSON file with data to run the optimization. The output file is saved in the specified output folder with the name 
    'dispatch_input_file_<site>.json'.

Notes: The script requires an optimization solver compatible with Pyomo. Default is MOSEK.

"""

import json
import pandas as pd
import numpy as np

import sys
import os

from shipp.components import Storage, Production
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo

from aux_forecast import get_p_min_vec

name_solver = 'mosek'

n = 24*2
nt = 8600
rel = 0.99
p_min = 10
p_max = 100
n_hist = 168
duration = 0
extra_power = 0
dt = 1
folder_in = 'data/'
folder_out = ''
filename = '../data/sites.csv'

if len(sys.argv) > 1:
    folder_in = sys.argv[1]
    assert folder_in[-1] == "/"
    assert os.path.isdir(folder_in)

if len(sys.argv) > 2:
    folder_out = sys.argv[2]
    assert folder_out[-1] == "/"
    assert os.path.isdir(folder_out)

if len(sys.argv) > 3:
    filename = sys.argv[3]

if len(sys.argv) > 4:
    site = sys.argv[4]

if len(sys.argv) > 5:
    n = int(sys.argv[5])
    print('n =', n)

if len(sys.argv) > 6:
    nt = int(sys.argv[6])
    print('nt =', nt)

if len(sys.argv) > 7:
    rel = float(sys.argv[7])
    print('rel =', rel)

if len(sys.argv) > 8:
    p_min = float(sys.argv[8])
    print('p_min =', p_min)

if len(sys.argv) > 9:
    n_hist = int(sys.argv[9])
    print('n_hist =', n_hist)

if len(sys.argv) > 10:
    duration = float(sys.argv[10])
    print('duration =', duration)

if len(sys.argv) > 11:
    extra_power = float(sys.argv[11])
    print('extra_power =', extra_power)
            
eta1 = 0.85
p_cap1 = 10.0
e_cap1 = 200
e_start = e_cap1/2


file_price_root = folder_in + "dam_price_entsoe_2019_{}.json"
file_windpower_root_vec = [folder_in + "mars_ptf_2019_60h_v2_{}_windpower.json",
                           folder_in + "mars_ptf_2019_60h_v2_{}_windpower_1.json",
                            folder_in + "mars_ptf_2019_60h_v2_{}_windpower_2.json" ]

print(site)

data = pd.read_csv(filename)
bz_vec = data['Bidding zone']
name_upper_vec = data['Name']
# Find the index in name_upper_vec that matches the site
try:
    index = name_upper_vec[name_upper_vec == site.upper()].index[0]
    bz = bz_vec[index]
    name = name_upper_vec[index]
except IndexError:
    raise ValueError(f"Site '{site}' not found in the dataset.")

# Extract price, wind power observation and forecast data

# Check price data
file_price = file_price_root.format(bz) 
print(file_price)

assert os.path.exists(file_price)
with open(file_price) as f:
    data_price = json.load(f)
price = data_price['price [EUR/MWh]']
assert not np.isnan(sum(price))

# Check wind power data
file_windpower_vec = []
cnt = 0
for file_windpower_root in file_windpower_root_vec:
    file_windpower = file_windpower_root.format(name.lower())

    assert os.path.exists(file_windpower)
    with open(file_windpower) as f:
        data_windpower = json.load(f)
    
    windpower_for = data_windpower['windpower forecast']
    if cnt == 0:
        # windpower_obs = data_windpower['windpower observations']
        windpower_obs = np.array([windpower_for[i][0][0] for i in range(len(windpower_for))])
        assert not np.isnan(sum(windpower_obs))
        assert p_min <= np.mean(windpower_obs)
        nt_max  = min(len(price), len(windpower_obs))
        assert n + nt <= nt_max 
        

        # Check match between parameters  
    
    n_file = len(windpower_for[0][0])
    assert n <= n_file
    
    

    file_windpower_vec.append(file_windpower)
    cnt+=1

# windpower_obs = np.array([forecast_vec[0][i][0][0] for i in range(len(forecast_ptf))])


# Calculation of storage capacity
if duration == 0: # Find optimal storage size if duration parameter not included
    stor = Storage(e_cap = None,  p_cap = None, eff_in = 1.0, eff_out = eta1, 
                   e_cost = 2*150*1e3, p_cost = 2*175*1e3)
    stor_null = Storage(e_cap = 0,  p_cap = 0, eff_in = 1.0, eff_out = 1.0)

    # Generate storage sizing automatically
    #price_ts = TimeSeries([5 for _ in range(n_max)], dt)
    price_ts = TimeSeries(price[:nt_max], dt)
    prod_wind = Production(TimeSeries(windpower_obs[:nt_max], dt), p_cost = 0)
    prod_pv = Production(TimeSeries([0 for _ in range(nt_max)], dt), p_cost = 0)

    p_min_vec = get_p_min_vec(p_min, windpower_obs[:nt_max], rel)

    os_res = solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor, stor_null, 0.03, 20, p_min_vec, p_max, nt_max, name_solver = name_solver)

    p_cap1 = os_res.storage_list[0].p_cap
    e_cap1 = os_res.storage_list[0].e_cap
    e_start = os_res.storage_e[0].data[0]    

else:
    p_cap1 = p_min + extra_power
    e_cap1 = duration*p_min
    e_start = 0.5*e_cap1

# Output data
file_input = folder_out + 'dispatch_input_file_'+ name.lower() + '.json'

with open(file_input, 'w', encoding = 'utf-8') as f:
    json.dump({'file_price': file_price, 'file_windpower': file_windpower_vec,
                'p_min': p_min, 'p_max': p_max, 'e_start': e_start, 
                'n': n, 'nt': nt, 'dt': dt, 'eta': eta1, 'p_cap': p_cap1, 'e_cap':e_cap1, 'rel':rel, 'n_hist': n_hist}, f, ensure_ascii=False, indent = 4)

print('Input files created:', file_input)