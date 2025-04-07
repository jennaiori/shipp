"""
This script generates JSON files containing day-ahead market (DAM) prices for specified bidding zones 
using data retrieved from the ENTSO-E API. The script reads input data from a CSV file, processes the 
prices to handle invalid or missing values, and saves the results in the specified output folder.

Arguments:
    output_folder (str): The directory where the JSON files will be saved.
    input_csv_file (str): The path to the CSV file containing bidding zone information.
    token_file (str): The path to the file containing the ENTSO-E API token.

Usage:
    python create_price_files.py [output_folder] [input_csv_file] [token_file]

Output:
    JSON files for each unique bidding zone in the input CSV file. Each JSON file contains:
        'price [EUR/MWh]': List of processed DAM prices.
        'bidding zone': The bidding zone code.
        'date_start': Start date of the data (fixed as '2019-01-01').
        'date_end': End date of the data (fixed as '2019-12-31').
        'source': Data source ('ENTSO-E Day-ahead price').
"""

import json
import pandas as pd
import numpy as np

import sys
import os

from shipp.io_functions import api_request_entsoe, replace_nan_with_mean

filename = 'data/sites.csv'
folder_out = 'data/'
token_filename = 'data/token_entsoe.txt'

if len(sys.argv) > 1:
    folder_out = sys.argv[1]
    assert folder_out[-1] == "/"
    assert os.path.isdir(folder_out)

if len(sys.argv) > 2:
    filename = sys.argv[2]

if len(sys.argv) > 3:
    token_filename = sys.argv[3]



data = pd.read_csv(filename)

bz_vec = data['Bidding zone']
date_start = '2019-01-01'
date_end = '2019-12-31'


file_price_root = folder_out + "dam_price_entsoe_2019_{}.json"

with open(token_filename, 'r') as file:
    token_entsoe = file.read().strip()

for bz in bz_vec.unique():
    print(bz)
    price = api_request_entsoe(token_entsoe, date_start, date_end, bz)
    
    # #  Remove any invalid data
    if min(price)<0:
        price = price - min(price) +0.5
    
    if np.isnan(sum(price)):
        price = replace_nan_with_mean(price)



    file_price = file_price_root.format(bz)
    with open(file_price, 'w', encoding = 'utf-8') as f:
        json.dump({'price [EUR/MWh]': price.tolist(), 'bidding zone': bz, 
                   'date_start': date_start, 'date_end': date_end, 
                   'source': 'ENTSO-E Day-ahead price'}, f, ensure_ascii=False, indent = 4)

    