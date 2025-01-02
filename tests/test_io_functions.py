import pytest
import requests
import json
from unittest.mock import patch
import numpy as np
import pandas as pd 
from shipp.io_functions import api_request_rninja, api_request_entsoe, get_power_price_data, save_power_price_to_json, get_power_price_from_json
import os

# Mock response data
mock_response_data = {
    'data': 
        {
        '1672531200000': {'electricity': 1000,  'wind_speed': 5}, 
        '1672534800000': {'electricity': 1500, 'wind_speed': 6}, 
        '1672538400000': {'electricity': 2000, 'wind_speed': 7}    
        }
}

@pytest.fixture
def mock_requests_get():
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = json.dumps(mock_response_data)
        yield mock_get

def test_api_request_rninja_success(mock_requests_get):

    token = 'valid_token'
    latitude = 52.0
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1500
    height = 100
    turbine = 'Vestas V164 8000'

    data_wind, data_power = api_request_rninja(
        token, latitude, longitude, date_start, date_end, capacity, height, turbine
    )

    assert np.array_equal(data_wind, np.array([5.0, 6.0, 7.0]))
    assert np.array_equal(data_power, np.array([1.0, 1.5, 2.0]))

def test_api_request_rninja_invalid_token(mock_requests_get):
    mock_requests_get.return_value.status_code = 401
    mock_requests_get.return_value.text = 'Unauthorized'

    token = 'invalid_token'
    latitude = 52.0
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 100
    turbine = 'Vestas V90'

    with pytest.raises(Exception, match='Error while requesting data from renewables.ninja'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_response(mock_requests_get):
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.text = 'Invalid JSON'

    token = 'valid_token'
    latitude = 52.0
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 100
    turbine = 'Vestas V90'

    with pytest.raises(Exception, match='Error while requesting data from renewables.ninja'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_latitude():
    token = 'valid_token'
    latitude = 'invalid_latitude'
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 100
    turbine = 'Vestas V90'

    with pytest.raises(ValueError, match='Latitude must be a float or int.'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_longitude():
    token = 'valid_token'
    latitude = 52.0
    longitude = 'invalid_longitude'
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 100
    turbine = 'Vestas V90'

    with pytest.raises(ValueError, match='Longitude must be a float or int.'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_date_format():
    token = 'valid_token'
    latitude = 52.0
    longitude = 4.0
    date_start = 'invalid_date'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 100
    turbine = 'Vestas V90'

    with pytest.raises(ValueError, match="Dates must be in 'YYYY-MM-DD' format."):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_capacity():
    token = 'valid_token'
    latitude = 52.0
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 'invalid_capacity'
    height = 100
    turbine = 'Vestas V90'

    with pytest.raises(ValueError, match='Capacity must be a float or int.'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_height():
    token = 'valid_token'
    latitude = 52.0
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 'invalid_height'
    turbine = 'Vestas V90'

    with pytest.raises(ValueError, match='Height must be a float or int.'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_turbine():
    token = 'valid_token'
    latitude = 52.0
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 100
    turbine = 12345

    with pytest.raises(ValueError, match='Turbine must be a string.'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )

def test_api_request_rninja_invalid_token_type():
    token = 12345
    latitude = 52.0
    longitude = 4.0
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    capacity = 1.5
    height = 100
    turbine = 'Vestas V90'

    with pytest.raises(ValueError, match='Token must be a string.'):
        api_request_rninja(
            token, latitude, longitude, date_start, date_end, capacity, height, turbine
        )
    return
   
@pytest.fixture
def mock_entsoe_client():
    with patch('shipp.io_functions.EntsoePandasClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.query_day_ahead_prices.return_value = pd.Series([50.0, 55.0, 60.0])
        yield mock_instance


def test_api_request_entsoe_success(mock_entsoe_client):
    token = 'valid_token'

    country_code = 'NL'
    date_start = '2019-01-01'
    date_end = '2019-01-02'

    prices = api_request_entsoe(token, date_start, date_end, country_code)

    assert np.array_equal(prices, np.array([50.0, 55.0, 60.0]))


def test_api_request_entsoe_invalid_token(mock_entsoe_client):
    token = 12345
    country_code = 'NL'
    date_start = '2023-01-01'
    date_end = '2023-01-02'

    with pytest.raises(ValueError, match='Token must be a string.'):
        api_request_entsoe(token, date_start, date_end, country_code)

def test_api_request_entsoe_invalid_country_code(mock_entsoe_client):
    token = 'valid_token'
    country_code = 12345
    date_start = '2023-01-01'
    date_end = '2023-01-02'

    with pytest.raises(ValueError, match='Country code must be a string.'):
        api_request_entsoe(token, date_start, date_end, country_code)

def test_api_request_entsoe_invalid_date_format(mock_entsoe_client):
    token = 'valid_token'
    country_code = 'NL'
    date_start = 'invalid_date'
    date_end = '2023-01-02'

    with pytest.raises(ValueError, match="Dates must be in 'YYYY-MM-DD' format."):
        api_request_entsoe(token, date_start, date_end, country_code)

def test_get_wind_price_data_success(mock_requests_get, mock_entsoe_client):
    token = 'valid_token'   
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    lattitude = 52.0
    longitude = 4.0
    capacity = 8000
    height = 164
    turbine = 'Vestas V164 8000'
    country_code = 'NL'

    data_power, data_price = get_power_price_data(token, token, date_start, date_end, 
                                                lattitude, longitude, capacity, 
                                                height, turbine, country_code)
    
    assert np.array_equal(data_power, np.array([1.0, 1.5, 2.0]))
    assert np.array_equal(data_price, np.array([50.0, 55.0, 60.0]))


def test_get_wind_price_data_invalid_token(mock_requests_get, mock_entsoe_client):
    token = 12345

    date_start = '2023-01-01'
    date_end = '2023-01-02'
    lattitude = 52.0
    longitude = 4.0
    capacity = 8000
    height = 164
    turbine = 'Vestas V164 8000'
    country_code = 'NL'

    with pytest.raises(ValueError, match='Token must be a string.'):
        get_power_price_data(token, token, date_start, date_end, lattitude, 
                            longitude, capacity, height, turbine, country_code)

def test_get_wind_price_data_invalid_country_code(mock_requests_get, mock_entsoe_client):
    token = 'valid_token'
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    lattitude = 52.0
    longitude = 4.0
    capacity = 8000
    height = 164
    turbine = 'Vestas V164 8000'
    country_code = 12345

    with pytest.raises(ValueError, match='Country code must be a string.'):
        get_power_price_data(token, token, date_start, date_end,lattitude, 
                            longitude, capacity, height, turbine, country_code)

def test_get_wind_price_data_invalid_date_format(mock_requests_get, mock_entsoe_client):
    token = 'valid_token'
    date_start = 'invalid_date'
    date_end = '2023-01-02'
    lattitude = 52.0
    longitude = 4.0
    capacity = 8000
    height = 164
    turbine = 'Vestas V164 8000'
    country_code = 'NL'

    with pytest.raises(ValueError, match="Dates must be in 'YYYY-MM-DD' format."):
         get_power_price_data(token, token, date_start, date_end, lattitude, 
                            longitude, capacity, height, turbine, country_code)

def test_get_wind_price_data_invalid_latitude(mock_requests_get, mock_entsoe_client):
    token = 'valid_token'
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    lattitude = 'invalid_latitude'

    longitude = 4.0
    capacity = 8000
    height = 164
    turbine = 'Vestas V164 8000'
    country_code = 'NL'

    with pytest.raises(ValueError, match='Latitude must be a float or int.'):
        get_power_price_data(token, token, date_start, date_end, lattitude, 
                            longitude, capacity, height, turbine, country_code)
        
def test_get_wind_price_data_invalid_longitude(mock_requests_get, mock_entsoe_client):
    token = 'valid_token'       
    date_start = '2023-01-01'
    date_end = '2023-01-02'
    lattitude = 52.0
    longitude = 'invalid_longitude'
    capacity = 8000
    height = 164
    turbine = 'Vestas V164 8000'
    country_code = 'NL'

    with pytest.raises(ValueError, match='Longitude must be a float or int.'):
        get_power_price_data(token, token, date_start, date_end, lattitude, 
                            longitude, capacity, height, turbine, country_code)
        


def test_save_power_price_to_json():
    # Save data to JSON
    filename = 'test_data.json'
    data_power = np.array([100, 200, 300])
    data_price = np.array([50, 60, 70])

    save_power_price_to_json(filename, data_power, data_price)

    # Check if file is created
    assert os.path.exists(filename)

    # Load data from JSON to verify
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Convert lists back to numpy arrays
    loaded_data_power = np.array(data['power'])
    loaded_data_price = np.array(data['price'])

    # Verify the data
    np.testing.assert_array_equal(loaded_data_power, data_power)
    np.testing.assert_array_equal(loaded_data_price, data_price)

    #Clean up the test file.
    if os.path.exists(filename):
        os.remove(filename)

def test_save_power_price_to_json_invalid_filename():
    filename = 'test_data.json'
    data_power = np.array([100, 200, 300])
    data_price = np.array([50, 60, 70])

    with pytest.raises(ValueError):
        save_power_price_to_json(123, data_power, data_price)

def test_save_power_price_to_json_invalid_data():
    filename = 'test_data.json'
    data_power = np.array([100, 200, 300])
    data_price = np.array([50, 60, 70])

    with pytest.raises(ValueError):
        save_power_price_to_json(filename, [100, 200, 300], data_price)
    with pytest.raises(ValueError):
        save_power_price_to_json(filename, data_power, [50, 60, 70])

def test_get_power_price_from_json():
    filename = 'test_data.json'
    with open(filename, 'w') as file:
        file.write('{"power": [100, 200, 300], "price":  [50, 60, 70]}')
    # Load data using the function
    loaded_data_power, loaded_data_price = get_power_price_from_json(filename)

    # Verify the data
    np.testing.assert_array_equal(loaded_data_power, [100, 200, 300])
    np.testing.assert_array_equal(loaded_data_price,  [50, 60, 70])

    #Clean up the test file.
    if os.path.exists(filename):
        os.remove(filename)

def test_get_power_price_from_json_invalid_filename():
    """Test get_power_price_from_json with invalid filename."""
    with pytest.raises(FileNotFoundError):
        get_power_price_from_json('non_existent_file.json')

def test_get_power_price_from_json_invalid_data():
    filename = 'test_data.json'

    # Create a malformed JSON file
    with open(filename, 'w') as file:
        file.write('{"power": [100, 200, 300], "price": [100, a, 300]}')

    with pytest.raises(ValueError):
        get_power_price_from_json(filename)
    
    #Clean up the test file.
    if os.path.exists(filename):
        os.remove(filename)


