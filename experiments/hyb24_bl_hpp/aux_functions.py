'''
    Functions:
        power_calc: Calculate power production from wind speed.
        plot_xcorr: Plot cross-correlation between two signals.


'''

import sys
sys.path.append('../../')

from shipp.kernel_pyomo import solve_lp_pyomo
from shipp.components import Storage, Production, TimeSeries

import numpy as np
import numpy_financial as npf
import pandas as pd
from scipy.stats import pearsonr
from scipy import signal
import math
import requests
import json
import time
from entsoe import EntsoePandasClient


from io import StringIO

def extract_wind_data(filename, token):
    data_wind  = []
    power_wind = []
    
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()
    # Send token header with each request
    s.headers = {'Authorization': 'Token ' + token}

    url = api_base + 'data/wind'

    df = pd.read_csv(filename)

    m = len(df)

    success = True

    for i in range(m):
        args = {
            'lat': df['Lat'][i],
            'lon': df['Long'][i],
            'date_from': df['Date from'][i],
            'date_to': df['Date to'][i],
            'capacity': df['Wind Farm Capacity [kW]'][i],
            'height': df[ 'Hub height [m]'][i],
            'turbine': df['Turbine name'][i],
            'format': 'json',
            'raw': 'true'
        }

        r = s.get(url, params=args)

        
        try:
            parsed_response = json.loads(r.text)
            data = pd.read_json(StringIO(json.dumps(parsed_response['data'])), orient='index')
            data_wind.append(np.array(data['wind_speed']))
            power_wind.append(np.array(data['electricity'])*1e-3)  
        except:
            print(r.text)
            success = False
            break
        
        time.sleep(0.8) 

    if success:
        print('{} data sets extracted succesfully from renewables.ninja.'.format(m))
    else:
        print('Error while loading data sets from renewables.ninja')

    return data_wind, power_wind


def postprocess_price_data(filename):
    rate_eur2usd = 1.09
    
    df = pd.read_csv(filename)
    
    m = len(df)

    price = []

    for i in range(m):
        data = pd.read_csv(df['Price file'][i], header = 0)
        price_og = data["Day-ahead Price [EUR/MWh]"].values * rate_eur2usd
        
        # Remove NaN data
        for i in range(0, len(price_og)):
            if math.isnan(price_og[i]):
                if i<(len(price_og)-1) and not math.isnan(price_og[i+1]):
                    price_og[i] = 0.5*(price_og[i-1] + price_og[i+1])
                else:
                    price_og[i] = (price_og[i-1])
   
        price_og_lowpass = butter_lowpass_filter(price_og, 0.15, 1) #0.15
        price_og_highpass = butter_highpass_filter(price_og_lowpass, 0.006, 1)
        
        price.append(np.maximum(1, (price_og_highpass + np.mean(price_og_lowpass))))
    
    
    return price


def extract_day_ahead_price(filename, token):
    # Extract data from input file
    df = pd.read_csv(filename)
    arr = df[['Date from', 'Date to', 'Bidding zone']].values

    arr_str = arr.astype(str)

    # Retrieve unique values
    unique_val, unique_ind = np.unique(arr_str, axis = 0, return_inverse=True)

    rate_eur2usd = 1.09
    unique_price = []
    client = EntsoePandasClient(api_key=token)

    for i in range(len(unique_val)):
        date_to = unique_val[i][0].replace('-', '')
        date_from = unique_val[i][1].replace('-', '')
        country_code = unique_val[i][2]

        start = pd.Timestamp(date_to, tz='Europe/Brussels')
        end = pd.Timestamp(date_from, tz='Europe/Brussels')

        print('Entso-e query:', country_code, date_to, date_from)

        ts = client.query_day_ahead_prices(country_code, start=start, end=end)
        
        price_og = ts.to_list()
        
        # Remove NaN data
        for j in range(0, len(price_og)):
            if math.isnan(price_og[j]):
                if i<(len(price_og)-1) and not math.isnan(price_og[j+1]):
                    price_og[j] = 0.5*(price_og[j-1] + price_og[j+1])
                else:
                    price_og[j] = (price_og[j-1])

        price_og_lowpass = butter_lowpass_filter(price_og, 0.15, 1) #0.15
        price_og_highpass = butter_highpass_filter(price_og_lowpass, 0.006, 1)
        price_filtered = np.maximum(1, (price_og_highpass + np.mean(price_og_lowpass))) *rate_eur2usd

        unique_price.append(price_filtered)

    unique_price = np.array(unique_price)

    price = unique_price[unique_ind]

    return price


def run_site_comparison(power_wind, price, n, dt, percent_bl, discount_rate,
                         n_year, bl_vec, stor_sts_a, stor_sts_c, stor_lts, 
                         pyo_solver, verbose = True):
    os_vec = []
    npv_vec = []
    irr_vec = []
    p_st_vec = []
    capex_vec = []
    rev_vec = []
    increase_rev_vec = []
    bl_nrg_vec = []
    p_range_vec = []
    corr_price_vec = []
    mean_price2 = []
    max_e_vec = []
    max_p_vec = []
    index_i = []
    wsp_vec = []
    sts_costs = []
    mean_p_vec = []

    m = len(power_wind)

    if verbose:
        print('Site\tBL [MW]\tRevenue [kUSD]\tRev incr.\tP_cap1/E_cap1\tP_cap2/E_cap2\tCost BL [M.USD]')

    for stor_batt in [stor_sts_a, stor_sts_c]:
        for p_min in bl_vec:
            for i in range(m):
                power_pv_ts = TimeSeries([0 for _ in range(n)], dt)
                power_wind_ts = TimeSeries(power_wind[i][:n], dt)
                price_ts = TimeSeries(price[i][:n], dt)
                prod_pv = Production(power_pv_ts, 0)
                prod_wind = Production(power_wind_ts, 0)
                
                # p_min = np.quantile(power_wind[i][:n], quantile)
                # p_min = 0.15*np.mean(power_wind[i][:n])

                power = np.array(power_wind[i][:n])
                
                p_min_vec = get_p_min_vec(p_min, power, percent_min=percent_bl)
                p_max = max(power_wind[i][:n])
                
                # os =  solve_lp_sparse(price_ts, prod_wind, prod_pv, stor_batt, stor_lts, 
                #             discount_rate, n_year, p_min_vec, p_max, n)
                
                os =  solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_batt, 
                                     stor_lts, discount_rate, n_year, 
                                     p_min_vec, p_max, n, pyo_solver)
                
                rev_res_only = 365 * 24 / n * np.dot(price[i][:n], np.minimum(power, p_max))*dt

                


                p_st, p_lt = get_percent(os, power[:n], p_min, p_min_vec)
                storage_capex = os.storage_list[0].get_tot_costs() + os.storage_list[1].get_tot_costs()
                a_rev = 365 * 24 / n * np.dot(price[i][:n], os.storage_p[0].data[:n] + os.storage_p[1].data[:n])*dt
                
                cash_flow = [-storage_capex]
                for _ in range(1,n_year):
                    cash_flow.append(a_rev)

                a_npv = npf.npv(discount_rate, cash_flow) * 1e-6

                if verbose:
                    print('{}\t{:.1f}\t{:.1f}\t\t{:.2f}%\t\t{:.2f}/{:.2f}\t{:.2f}/{:.2f}\t{:.2f}'.format(i, p_min, 
                                                                                        os.revenue*1e-3, 100*(os.revenue/rev_res_only-1), 
                                                                                        os.storage_list[0].p_cap, 
                                                                                        os.storage_list[0].e_cap, os.storage_list[1].p_cap, 
                                                                                        os.storage_list[1].e_cap, -a_npv))

                p_st_vec.append(p_st)
                npv_vec.append(a_npv)
                capex_vec.append(storage_capex*1e-6)
                rev_vec.append(a_rev*1e-3)
                irr_vec.append(os.irr)
                increase_rev_vec.append(100*(os.revenue/rev_res_only-1))

                nrg = 100*sum([p_min_tmp-p if p<= p_min_tmp else 0 for p, p_min_tmp in zip(power_wind[i][:n], p_min_vec)])/sum(power_wind[i])
                bl_nrg_vec.append(nrg)

                corr_price = pearsonr(power_wind[i][:n]/np.std(power_wind[i][:n]), price[i][:n]/np.std(price[i][:n])).statistic

                os_vec.append(os)
                p_range_vec.append(p_min)
                mean_price2.append(np.mean(price[i][:n]))
                corr_price_vec.append(corr_price)

                index_i.append(i)
                sts_costs.append(stor_batt.e_cost)
                mean_p_vec.append(np.mean(power_wind[i]))


    p_range_vec = np.array(p_range_vec)
    corr_price_vec = np.array(corr_price_vec)
    npv_vec = np.array(npv_vec)
    bl_nrg_vec = np.array(bl_nrg_vec)
    mean_p_vec = np.array(mean_p_vec)
    sts_costs = np.array(sts_costs)
    
    
    data_site_comparison = [p_st_vec,
                    npv_vec,
                    capex_vec,
                    rev_vec,
                    irr_vec,
                    increase_rev_vec,
                    os_vec,
                    p_range_vec,
                    mean_price2,
                    corr_price_vec,
                    max_e_vec,
                    max_p_vec,
                    index_i,
                    wsp_vec,
                    sts_costs,
                    mean_p_vec,
                    bl_nrg_vec]



    return data_site_comparison


def run_cost_comparison(power_wind, price, n, dt, percent_bl, discount_rate,
                        n_year, bl_reference, stor_sts, stor_lts, e_cost_range, 
                        p_cost_range, pyo_solver, verbose = True):
    a_npv_mat = []
    p_st_mat = []

    e_cap_lt_mat = []
    p_cap_lt_mat = []
    e_cap_st_mat = []
    p_cap_st_mat = []
    a_rev_mat = []

    m = len(power_wind)

    for idx in range(m):
        if verbose:
            print(idx)
        a_npv_mat_tmp = np.zeros((len(p_cost_range), len(e_cost_range)))
        p_st_mat_tmp = np.zeros((len(p_cost_range), len(e_cost_range)))

        e_cap_lt_mat_tmp = np.zeros((len(p_cost_range), len(e_cost_range)))
        p_cap_lt_mat_tmp = np.zeros((len(p_cost_range), len(e_cost_range)))
        e_cap_st_mat_tmp = np.zeros((len(p_cost_range), len(e_cost_range)))
        p_cap_st_mat_tmp = np.zeros((len(p_cost_range), len(e_cost_range)))
        a_rev_mat_tmp = np.zeros((len(p_cost_range), len(e_cost_range)))

        prod_wind = Production(TimeSeries(power_wind[idx][:n], dt), 0)
        prod_pv = Production(TimeSeries([ 0  for _ in range(n)], dt), 0)
        
        price_ts = TimeSeries(price[idx][:n], dt)

        p_min = bl_reference
        p_max = max(prod_wind.power.data[:n])
        p_min_vec = get_p_min_vec(p_min, prod_wind.power.data[:n])

        for i in range(len(p_cost_range)):
            for j in range(len(e_cost_range)):

                stor_st_tmp = Storage(e_cap = None, p_cap = None, eff_in = 1, eff_out = stor_sts.eff_out,
                                    e_cost= 2*e_cost_range[j], p_cost= stor_sts.p_cost)
                stor_lt_tmp = Storage(e_cap = None, p_cap = None, eff_in = 1, eff_out = stor_lts.eff_out, 
                                    e_cost= stor_lts.e_cost, p_cost= p_cost_range[i])


                # os =  solve_lp_sparse(price_ts, prod_wind, prod_pv, stor_st_tmp, stor_lt_tmp, 
                #         discount_rate, n_year, p_min_vec, p_max, n)

                os =  solve_lp_pyomo(price_ts, prod_wind, prod_pv, stor_st_tmp, stor_lt_tmp, 
                        discount_rate, n_year, p_min_vec, p_max, n, pyo_solver)

                p_st, p_lt = get_percent(os, prod_wind.power.data[:n], p_min, p_min_vec)

                storage_capex = os.storage_list[0].get_tot_costs() + os.storage_list[1].get_tot_costs()
                res_capex = os.production_list[0].get_tot_costs() + os.production_list[1].get_tot_costs()
                
                a_rev = 365 * 24 / n * np.dot(price_ts.data[:n], os.storage_p[0].data[:n] + os.storage_p[1].data[:n])*dt

                rev_res_only = 365 * 24 / n * np.dot(price_ts.data[:n], np.minimum(prod_wind.power.data[:n], p_max))*dt

                cash_flow = [-storage_capex]
                for _ in range(1,n_year):
                    cash_flow.append(a_rev)

                a_npv = npf.npv(discount_rate, cash_flow) * 1e-6

                a_npv_mat_tmp[i][j] = a_npv
                p_st_mat_tmp[i][j] = p_st
                e_cap_lt_mat_tmp[i][j] = os.storage_list[1].e_cap
                p_cap_lt_mat_tmp[i][j] = os.storage_list[1].p_cap
                e_cap_st_mat_tmp[i][j] = os.storage_list[0].e_cap
                p_cap_st_mat_tmp[i][j] = os.storage_list[0].p_cap
                a_rev_mat_tmp[i][j] = a_rev


        a_npv_mat.append(a_npv_mat_tmp)
        p_st_mat.append(p_st_mat_tmp)
        e_cap_lt_mat.append(e_cap_lt_mat_tmp)
        p_cap_lt_mat.append(p_cap_lt_mat_tmp)
        e_cap_st_mat.append(e_cap_st_mat_tmp)
        p_cap_st_mat.append(p_cap_st_mat_tmp)
        a_rev_mat.append(a_rev_mat_tmp)


    data_cost_comparison = [e_cost_range,
                            p_cost_range,
                            a_npv_mat,
                            p_st_mat,
                            e_cap_lt_mat,
                            p_cap_lt_mat,
                            e_cap_st_mat,
                            p_cap_st_mat,
                            a_rev_mat]



    return data_cost_comparison

def get_percent(os, power, p_min, p_min_vec):

    bl = (p_min_vec[power<p_min] - power[power<p_min])*p_min_vec[power<p_min]/p_min

    bl_lt = os.storage_p[1].data[power<p_min]*p_min_vec[power<p_min]/p_min
    bl_st = os.storage_p[0].data[power<p_min]*p_min_vec[power<p_min]/p_min

    bl_lt = np.minimum(bl_lt, bl)
    bl_st = np.maximum(np.minimum(bl_st, bl-bl_lt), 0)

    return sum(bl_st)/sum(bl)*100, sum(bl_lt)/sum(bl)*100

def find_extreme_event(power, p_min_vec, n):
    windows_vec = []
    window_tmp = []

    for i in range(n):
        p = power[i]
        p_min_tmp = p_min_vec[i]
        if p<p_min_tmp:
            if i==0:
                window_tmp.append(i)
            else:
                if i-1 in window_tmp:
                    window_tmp.append(i)
                else:
                    windows_vec.append(window_tmp)
                    window_tmp = [i]

    max_len = 0
    max_nrg = 0
    max_power = 0
    for i in range(len(windows_vec)):
        # plt.plot(windows_vec[i], power[windows_vec[i]], 'k')
        # plt.plot(windows_vec[i], p_min_vec[windows_vec[i]], 'r')
        max_len = max(max_len, len(windows_vec[i]))
        max_nrg = max(max_nrg, np.sum(p_min_vec[windows_vec[i]] - power[windows_vec[i]])*dt)
        if len(windows_vec[i])>0:
            max_power = max(max_power, max(p_min_vec[windows_vec[i]] - power[windows_vec[i]]))

    return max_nrg, max_power


# Find vector P_min_vec so that P>=P_min 99% of the time

def get_p_min_vec(p_min, data, percent_min = 0.99, return_len = False):
    len_continue_operation = 0
    len_max = 240

    vec_99_pc = np.zeros_like(data)
    percent = sum(vec_99_pc)/len(vec_99_pc)

    while percent<percent_min and len_continue_operation<len_max:
        vec_99_pc = np.zeros_like(data)
        for i in range(len(data)):
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

    # print(len_continue_operation-1, percent)
    assert (sum(vec_99_pc)/len(vec_99_pc))>= percent_min
    if len_continue_operation>= len_max-1:
        print('Warning get_p_min_vec: maximum length reached')

    if return_len:
        return len_continue_operation-1
    else:
        return p_min * vec_99_pc


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def load_csv_data(files, name_data, n_header = 0, mult = 1, n_cut = 0):

    res = np.array([])
    for filename in files:
        data = pd.read_csv(filename, header = n_header)
        res_tmp = data[name_data].values * mult
        
        for i in range(0, len(res_tmp)):
            if math.isnan(res_tmp[i]):
                if i<(len(res_tmp)-1):
                    res_tmp[i] = 0.5*(res_tmp[i-1] + res_tmp[i+1])
                else:
                    res_tmp[i] = (res_tmp[i-1])
        res_tmp = res_tmp[:len(res_tmp)-n_cut]

        res = np.append(res,res_tmp)
    return res

def create_site_file(filename):
    # # Some code to write the input file for the data extraction
    lat = [52.715111,
            51.70278,
            52.36666694,
            54.036000,
            54.05000,
            54.667,
            55.69694,
            56.60000,
            55.06278,
            54.824806]
    long = [4.251000,
                3.07611,
                4.11666694,
                5.963000,
                7.03000,
                6.167,
                7.66917,
                11.21000,
                13.00472,
                13.861694]

    files_price = ["data/Day-ahead Prices_enstoe_2019.csv",
                "data/Day-ahead Prices_enstoe_2019.csv",
                "data/Day-ahead Prices_enstoe_2019.csv",
                "data/Day-ahead Prices_enstoe_2019.csv",
                "data/Day-ahead Prices_entsoe_2019_DE.csv",
                "data/Day-ahead Prices_entsoe_2019_DE.csv",
                "data/Day-ahead Prices_entsoe_2019_DK1.csv",
                "data/Day-ahead Prices_entsoe_2019_DK1.csv",
                "data/Day-ahead Prices_entsoe_2019_DK2.csv",
                "data/Day-ahead Prices_entsoe_2019_DE.csv"]

    list_zones = ['NL', 'NL', 'NL', 'NL', 'DE_LU', 'DE_LU', 'DK_1', 'DK_1', 'DK_2', 'DE_LU']

    turbine = ['Siemens SWT 4.0 130', 'Vestas V164 8000', 'Enercon E126 6500' ]
    hub_height = [80, 140, 135]
    rotor_radius = [130/2, 164/2, 126/2]
    rated_power = [4000, 8000, 6500]

    date_from = '2019-01-01'
    date_to = '2020-01-01'
    wind_farm_cap = 100000.0

    lat_vec = []
    long_vec = []
    capacity_vec =[]
    turbine_vec =[]
    rated_power_vec = []
    hub_height_vec = []
    rotor_radius_vec = []
    date_from_vec = []
    date_to_vec = []
    zones_vec = []
    price_file_vec = []

    for i in range(len(lat)):
        ## Wind Power
        for k in range(len(turbine)):
            lat_vec.append(lat[i])
            long_vec.append(long[i])
            capacity_vec.append(wind_farm_cap)
            turbine_vec.append(turbine[k])
            rated_power_vec.append(rated_power[k])
            hub_height_vec.append(hub_height[k])
            rotor_radius_vec.append(rotor_radius[k])
            date_from_vec.append(date_from)
            date_to_vec.append(date_to)
            zones_vec.append(list_zones[i])
            price_file_vec.append(files_price[i])

    df = pd.DataFrame({'Lat': lat_vec,'Long': long_vec,
                    'Wind Farm Capacity [kW]': capacity_vec,
                    'Turbine name': turbine_vec, 'Turbine Capacity [kW]': rated_power_vec,
                    'Hub height [m]': hub_height_vec, 'Rotor radius [m]': rotor_radius_vec,
                    'Date from': date_from_vec, 'Date to': date_to_vec,
                    'Bidding zone': zones_vec, 'Price file': price_file_vec})

    df.to_csv(filename)

    return    



