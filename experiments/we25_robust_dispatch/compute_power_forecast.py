'''
compute_power_forecast.py

This script generates a power forecast file from ERA5 and MARS JSON data files for a wind farm.
It assumes a simple power curve for the wind farm and includes options to generate a pessimistic forecast.

Arguments:
    folder_in (str): Path to the input folder
    folder_out (str): Path to the output folder
    windpower_parameters (str): Path to the wind power parameters JSON file
    forecast_parameters (str): Path to the forecast parameters JSON file
    site_name (str): Name of the site
    pessimism_factor (float): Pessimism factor for the forecast

Usage:
    python compute_power_forecast.py [folder_in] [folder_out] [windpower_parameters] [forecast_parameters] [site_name] [pessimism_factor]

Output:
    A JSON file containing the power forecast data is saved in the specified output folder.

Parameters:
    time_delta (int): The forecast is issued every `time_delta` hours.
    h_arma (int): Forecast length for the ARMA forecast (in hours).
    n_hist_arma (int): Number of time steps to consider for the ARMA forecast.
    h_obs_mix (int): Number of hours to mix forecast data with observations.
'''



import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import json
from aux_forecast import mix_signals, get_wf_power
from scipy.interpolate import interp1d


# Parameters

folder_in = 'data/'
folder_out = 'data/'
file_windpower_parameters = 'data/windpower_parameters.json'
file_forecast_parameters = 'data/forecast_parameters.json'

if len(sys.argv) > 1:
    folder_in = sys.argv[1]
    assert folder_in[-1] == "/"
    assert os.path.isdir(folder_in)

if len(sys.argv) > 2:
    folder_out = sys.argv[2]
    assert folder_out[-1] == "/"
    assert os.path.isdir(folder_out)

if len(sys.argv) > 4:
    file_windpower_parameters = sys.argv[3]
    file_forecast_parameters = sys.argv[4]


if len(sys.argv) > 5:
    site_name = sys.argv[5]
else:
    site_name = 'hkn'

if len(sys.argv) > 6:
    pessimism_factor = float(sys.argv[6])
else:
    pessimism_factor = 0

# Load parameters
with open(file_windpower_parameters) as f:
    windpower_parameters = json.load(f)

hub_height = windpower_parameters["hub height"]
shear_coef = windpower_parameters["shear coefficient"]

with open(file_forecast_parameters) as f:
    forecast_parameters = json.load(f)

h_arma = forecast_parameters["h_arma"]
n_hist_arma = forecast_parameters["n_hist_arma"]
h_obs_mix = forecast_parameters["h_obs_mix"]
sigma_start = forecast_parameters["sigma_start"]
sigma_end = forecast_parameters["sigma_end"]
sigma_index = forecast_parameters["sigma_index"]
#Correction factors between ERA5 and ECMWF
corr_factor_u = forecast_parameters["corr_factor_u"]
corr_offset_u = forecast_parameters["corr_offset_u"]
corr_factor_v = forecast_parameters["corr_factor_v"]
corr_offset_v = forecast_parameters["corr_offset_v"]


# Files for the wind speed data forecast
file_forecast = folder_in+'mars_ptf_2019_60h_v2_{}.json'.format(site_name)

#File for the wind speed data observations
file_obs = folder_in+'data_era5_2019_{}.json'.format(site_name)

print('Input files:')
print("Wind speed forecast:",file_forecast)
# print(file_era5_dec)
print("Wind speed observations:",file_obs)

# Load forecast data
with open(file_forecast, 'r') as f:
    data = json.load(f)

if 'u10' in data and 'v10' in data:
    u_data = data['u10']
    v_data = data['v10']
    ref_height = 10
   
elif 'u100' in data and 'v100' in data:
    u_data = data['u100']
    v_data = data['v100']
    ref_height = 100

else:
    print('Error with forecast data')
    print(data.keys())

correction_factor = (hub_height/ref_height)**shear_coef # factor to adjust the wind speed to find hub height speed
steps = data['steps'] # vector of lead-time for which data exists
m = len(u_data[0]) # number of scenarios

time = data['time'] 
time = np.array(time, dtype = 'datetime64[ns]')
time = time.astype('datetime64[h]')

time_delta = (time[1]-time[0]).astype('int')  # the forecast is issued every 12 hours


# Load ERA5 data
with open(file_obs, 'r') as f:
    data_obs = json.load(f)

u_obs = []
v_obs = []
time_obs = []
for i in range(len(data_obs['time'])):
    if ref_height == 10:
        u_obs.append(data_obs['u10'][i][0])
        v_obs.append(data_obs['v10'][i][0])
    elif ref_height == 100:
        u_obs.append(data_obs['u100'][i][0])
        v_obs.append(data_obs['v100'][i][0])
    time_obs.append(data_obs['time'][i])
    
time_obs = np.array(time_obs, dtype = 'datetime64[ns]')
time_obs = time_obs.astype('datetime64[h]')
wdir_obs = [ np.mod(180 + np.arctan2(u, v) * 180 / np.pi, 360) 
             for u,v in zip(u_obs, v_obs)] #see definition of the wind direction on ecmwf website
wsp_obs = [ correction_factor* np.sqrt(u**2 + v**2) for u,v in zip(u_obs, v_obs)] # windspeed is corrected with appropriate factor

windpower_obs = [get_wf_power(wsp_mag, wsp_dir, windpower_parameters) for wsp_mag, wsp_dir in zip(wsp_obs, wdir_obs)]

# Create object to identify the original (og) date for the start of the forecast
og_date_offset = 0 # 31*24
og_date = time_obs[og_date_offset].astype('int64') - time_obs[0].astype('int64')

n = steps[-1]+1
steps_new = range(n)
steps_cut = range(n-time_delta) 

pessimism_vector = pessimism_factor*np.linspace(sigma_start, sigma_end, n)
pessimism_vector[:sigma_index+1] = pessimism_factor*np.linspace(0, sigma_start, sigma_index+1)
pessimism_vector[sigma_index:] = pessimism_factor*np.linspace(sigma_start, sigma_end, len(pessimism_vector[sigma_index:]))

print("Scenarios:\t", m)
print('Start date:\t', time[0])
print("nt [days]\t", len(time)/2)
print("n [-]:\t\t", n)

windpower_for_data = []
wsp_for_data = []

for i in range(len(time)-2): #date of forecast issue in 12h time deltas
    for j in range(time_delta):
        windpower_for_ens = []
        wsp_for_ens = []

        current_date = time_delta*i+j+og_date
        
        for number in range(m):
            #Retrieve the forecast with correction to match era5 data
            u_tmp = [(u-corr_offset_u)/corr_factor_u for u in u_data[i][number]]
            v_tmp = [(v-corr_offset_v)/corr_factor_v for v in v_data[i][number]]

            wsp_tmp = [ correction_factor*np.sqrt(u**2 + v**2) for u,v in zip(u_tmp, v_tmp)]
            wdir_tmp = [ np.mod(180 + np.arctan2(u, v) * 180 / np.pi, 360) 
                for u,v in zip(u_tmp, v_tmp)]
            windpower_tmp = [get_wf_power(wsp_mag, wsp_dir, windpower_parameters) 
                        for wsp_mag, wsp_dir in zip(wsp_tmp, wdir_tmp)]
            
            # Interpolate the forecast on a finer time discretization
            interp_function_wsp = interp1d(steps, wsp_tmp, kind = 'cubic')
            interp_function_wdir = interp1d(steps, wdir_tmp, kind = 'cubic')
            
            wsp_tmp_new = interp_function_wsp(steps_new[j:])  
            wdir_tmp_new = interp_function_wdir(steps_new[j:])

            
            # Mix forecast signal with observation
            wsp_mixed = mix_signals(wsp_tmp_new[:h_obs_mix], wsp_obs[current_date:current_date+h_obs_mix], h_obs_mix)
            wdir_mixed = mix_signals(wdir_tmp_new[:h_obs_mix], wdir_obs[current_date:current_date+h_obs_mix], h_obs_mix)

            for k in range(h_obs_mix):
                wsp_tmp_new[k] = wsp_mixed[k]
                wdir_tmp_new[k] = wdir_mixed[k]

            # The pessimistic factor is only applied to the wind speed               
            for k in range(1, n-j):
                wsp_tmp_new[k] = wsp_tmp_new[k]*(1-pessimism_vector[k])

            windpower_tmp_new = [get_wf_power(wsp_mag, wsp_dir, windpower_parameters) 
                        for wsp_mag, wsp_dir in zip(wsp_tmp_new[:j-time_delta], wdir_tmp_new[:j-time_delta])]

            windpower_tmp_new[0] = windpower_obs[current_date] #ensure the 0 lead time is equal to the observation
            
           
            wsp_for_ens.append([wsp for wsp in wsp_tmp_new])
            windpower_for_ens.append(windpower_tmp_new)

        wsp_for_data.append(wsp_for_ens) 
        windpower_for_data.append(windpower_for_ens)
    
if pessimism_factor > 0 and pessimism_factor<1:
    worstcase_str = "_{:.0f}pc".format(pessimism_factor*100)
elif pessimism_factor>=1:
    worstcase_str = "_{:.0f}".format(pessimism_factor)
else:
    worstcase_str = ''
file_out = file_forecast.replace(folder_in, folder_out)
file_out = file_out.replace('.json', '_windpower'+worstcase_str+'.json')


output_data = {
    "time": time_obs[og_date:og_date+len(windpower_for_data)].astype('int64').tolist(),
    "steps": [s for s in steps_cut],
    "lat": data['lat'],
    "lon": data['lon'],
    "windpower observations": windpower_obs[og_date:],
    "windpower forecast": windpower_for_data,
    "wind speed observations": wsp_obs[og_date:],
    "wind speed forecast": wsp_for_data,
    "worstcase": pessimism_factor,
    "windpower_parameters": windpower_parameters,
    "forecast_parameters": forecast_parameters
}

with open(file_out, 'w') as outfile:
    json.dump(output_data, outfile, indent=4)

