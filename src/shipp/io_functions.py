import sys
import time
import math
import json
from io import StringIO

import requests
import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient

def replace_nan_with_mean(data: np.ndarray) -> np.ndarray:
    """
    Replace NaN values in the array with the mean value of the series.

    Params:
        data (np.ndarray): Input array with potential NaN values.
    Returns:
        np.ndarray: Array with NaN values replaced by the mean value of the series.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    mean_value = np.nanmean(data)
    data = np.where(np.isnan(data), mean_value, data)
    
    return data


def api_request_rninja(token: str, latitude: float, longitude: float, 
                       date_start: str, date_end: str, 
                       capacity : float = 8000, height: float = 164, 
                       turbine: str = 'Vestas V164 8000'
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Makes an API request to renewables.ninja to retrieve wind speed and power data.

    Params:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        date_start (str): Start date for the data in 'YYYY-MM-DD' format.
        date_end (str): End date for the data in 'YYYY-MM-DD' format.
        capacity (float): Capacity of the wind turbine/farm in kW.
        height (float): Height of the wind turbine in meters.
        turbine (str): Type of wind turbine.
        token (str): API token for authentication.
    Returns:
        data_wind (numpy.ndarray): Array of wind speed data.
        data_power (numpy.ndarray): Array of wind power data in MW.
    Raises:
        Exception: If there is an error during the API request
    """

    # Validate input parameters
    if not isinstance(latitude, (float, int)):
        raise ValueError("Latitude must be a float or int.")
    if not isinstance(longitude, (float, int)):
        raise ValueError("Longitude must be a float or int.")
    if not isinstance(date_start, str) or not isinstance(date_end, str):
        raise ValueError("Dates must be strings")
    try:
        pd.to_datetime(date_start, format='%Y-%m-%d')
        pd.to_datetime(date_end, format='%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")
    if not isinstance(capacity, (float, int)):
        raise ValueError("Capacity must be a float or int.")
    if not isinstance(height, (float, int)):
        raise ValueError("Height must be a float or int.")
    if not isinstance(turbine, str):
        raise ValueError("Turbine must be a string.")
    if not isinstance(token, str):
        raise ValueError("Token must be a string.")

    api_base = 'https://www.renewables.ninja/api/'

    session = requests.session()
    # Send token header with each request
    session.headers = {'Authorization': 'Token ' + token}
    url = api_base + 'data/wind'

    args = {
        'lat': latitude,
        'lon': longitude,
        'date_from': date_start,
        'date_to': date_end,
        'capacity': capacity,
        'height': height,
        'turbine': turbine,
        'format': 'json',
        'raw': 'true'
    }

    # Send request
    req = session.get(url, params=args)
    try:
        # Retrieve response as in json format
        parsed_response = json.loads(req.text)
        data = pd.read_json(StringIO(json.dumps(parsed_response['data'])), orient='index')
        
        # Extract wind speed and wind power
        data_wind = np.array(data['wind_speed'])
        data_power = np.array(data['electricity']) * 1e-3
    except Exception as e:
        # In case of exception, print the error text from the API
        print('Request text:', req.text)
        raise Exception('Error while requesting data from renewables.ninja') from e

    return data_wind, data_power

def api_request_entsoe(token: str, date_start: str, date_end: str, 
                       country_code: str = 'NL' ) -> np.ndarray:
    """
    Fetches day-ahead electricity prices from the ENTSO-E API for a specified country zone and date range.

    Params:
        token (str): The API token for authenticating with the ENTSO-E API.
        country_code (str): The country code for which to fetch the day-ahead prices (e.g., 'DE' for Germany).
        date_start (str): The start date for the data request in 'YYYY-MM-DD' format.
        date_end (str): The end date for the data request in 'YYYY-MM-DD' format.

    Returns:
        np.ndarray: An array of day-ahead electricity prices for the specified country and date range.

    Raises:
        Exception: If there is an error while requesting data from the ENTSO-E API.
    """

    # Validate input parameters
    if not isinstance(token, str):
        raise ValueError("Token must be a string.")
    if not isinstance(country_code, str):
        raise ValueError("Country code must be a string.")
    if not isinstance(date_end, str) or not isinstance(date_start, str):
        raise ValueError("Dates must be strings.")
    try:
        pd.to_datetime(date_end, format='%Y-%m-%d')
        pd.to_datetime(date_start, format='%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")

    client = EntsoePandasClient(api_key=token)

    start = pd.Timestamp(date_start.replace('-', ''), tz='Europe/Brussels')
    end = pd.Timestamp(date_end.replace('-', ''), tz='Europe/Brussels')

    # Send request to API
    res = client.query_day_ahead_prices(country_code, start=start, end=end)

    try:
        data_price_og = res.to_list()
        data_price = replace_nan_with_mean(np.array(data_price_og))
    except Exception as e:
        # In case of exception, print the error text from the API
        raise Exception('Error while requesting data from ENTSO-E') from e

    return data_price

def get_power_price_data(token_rninja: str, token_entsoe: str, date_start: str, date_end: str,
                        latitude: float, longitude: float, 
                        capacity : float = 8000, height: float = 164, 
                       turbine: str = 'Vestas V164 8000', 
                       country_code: str = 'NL') -> tuple[np.ndarray, np.ndarray]:
    
    """
    Fetches (wind) power and price data for a specified location and time period.
    Params:
        token (str): Authentication token for API requests.
        date_start (str): Start date for the data retrieval in 'YYYY-MM-DD' format.
        date_end (str): End date for the data retrieval in 'YYYY-MM-DD' format.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        capacity (float, optional): Capacity of the wind turbine in kW. Default is 8000.
        height (float, optional): Height of the wind turbine in meters. Default is 164.
        turbine (str, optional): Model of the wind turbine. Default is 'Vestas V164 8000'.
        country_code (str, optional): Country zone for price data. Default is 'NL'.
    Returns:
        data_power: Wind power data (in MW).
        data_price: Price data (in Eur).
    """

    _, data_power = api_request_rninja(token_rninja, latitude, longitude, date_start, 
                                       date_end, capacity, height, turbine)
    data_price = api_request_entsoe(token_entsoe, date_start, date_end, country_code)

    return data_power, data_price

def save_power_price_to_json(filename: str, data_power: np.array, data_price: np.array):
    """
    Save power and price data to a JSON file.

    Params:
        filename (str): Name of the file to save the data.
        data_power (np.array): Power data.
        data_price (np.array): Price data.
    """

    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")
    if not isinstance(data_power, np.ndarray) or not isinstance(data_price, np.ndarray):
        raise ValueError("Data must be numpy arrays.")

    data = {
        'power': data_power.tolist(),
        'price': data_price.tolist()
    }

    with open(filename, 'w') as file:
        json.dump(data, file)

def get_power_price_from_json(filename: str) -> tuple[np.array, np.array]:
    """
    Load power and price data from a JSON file.

    Params:
        filename (str): Name of the file to load the data from.
    Returns:
        data_power (np.array): Power data.
        data_price (np.array): Price data.
    """

    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")

    with open(filename, 'r') as file:
        data = json.load(file)

    data_power = np.array(data['power'])
    data_price = np.array(data['price'])

    return data_power, data_price
