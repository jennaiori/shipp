"""
Auxiliary functions for HPP eco-design.

Functions:
    to_iso3: normalize a country identifier to its ISO3 code.
"""

import os
import numpy as np
import pandas as pd
from shipp.timeseries import TimeSeries
from shipp.io_functions import api_request_rninja
import json
import requests

BETA_OBJ_DEFAULT = 1e-4
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cache')


### Country identifier conversion

COUNTRY_TO_ISO3 = {
    'NLD': ('NL', 'NETHERLANDS'),
    'DEU': ('DE', 'GERMANY'),
    'DNK': ('DK', 'DENMARK'),
    'BEL': ('BE', 'BELGIUM'),
    'FRA': ('FR', 'FRANCE'),
    'GBR': ('GB', 'UK', 'UNITED KINGDOM'),
    'NOR': ('NO', 'NORWAY'),
    'SWE': ('SE', 'SWEDEN'),
    'ESP': ('ES', 'SPAIN'),
    'PRT': ('PT', 'PORTUGAL'),
    'ITA': ('IT', 'ITALY'),
    'POL': ('PL', 'POLAND'),
    'AUT': ('AT', 'AUSTRIA'),
    'CHE': ('CH', 'SWITZERLAND'),
    'IRL': ('IE', 'IRELAND'),
}


def to_iso3(country):
    """Normalize a country identifier to its ISO3 code.

    Args:
        country (str): country identifier (ISO3, ISO2, or name).

    Returns:
        str: ISO3 code (e.g. 'NLD').

    Raises:
        TypeError: if country is not a string.
        ValueError: if the identifier is not recognized.
    """
    if not isinstance(country, str):
        raise TypeError(
            'country must be a string, got {}'.format(type(country).__name__))
    key = country.strip().upper()
    if key in COUNTRY_TO_ISO3:
        return key
    for iso3, aliases in COUNTRY_TO_ISO3.items():
        if key in aliases:
            return iso3
    raise ValueError(
        "Country '{}' not recognized. Extend COUNTRY_TO_ISO3 in "
        "wrapper/auxiliary_functions.py to add it.".format(country))


### Caching

def cache_path(cache_dir, resource, **kwargs):
    """Build a deterministic cache path from resource and arguments."""
    parts = [str(resource)]
    for key in sorted(kwargs):
        val = kwargs[key]
        if isinstance(val, (list, tuple)):
            val = '-'.join(str(v) for v in val)
        parts.append('{}-{}'.format(key, val))
    return os.path.join(cache_dir, '_'.join(parts) + '.json')

def read_cache(path):
    """Return (data, meta) if the cache file exists and is valid, else None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r') as f:
            payload = json.load(f)
        return payload['data'], payload.get('meta', {})
    except (json.JSONDecodeError, KeyError):
        return None

def write_cache(path, data, meta):
    """Write a cache file with the given data and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'data': list(data), 'meta': meta}, f, indent=4)


### Day-ahead market price loading from CSV

def load_price_from_csv(iso3, year, csv_path, cache_dir=DEFAULT_CACHE_DIR):
    """Load day-ahead prices from a .csv file for a given country and year.
    Expected columns are 'ISO3 Code', 'Datetime (UTC)', and 'Price (EUR/MWhe)'.
    Negative prices are clipped to zero.


    Args:
        iso3 (str): ISO3 country code (e.g. 'NLD').
        year (int): calendar year.
        csv_path (str): path to the CSV file.
        cache_dir (str): cache directory.

    Returns:
        TimeSeries: hourly prices [currency/MWh], dt=1.0.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if no rows match the given ISO3 and year.
        AssertionError: if the reindexed series contains NaN.
    """
    cache_file = cache_path(cache_dir, 'price', iso3=iso3, year=year,
                            source='csv')
    cached = read_cache(cache_file)
    if cached is not None:
        data, _ = cached
        return TimeSeries(np.array(data), 1.0)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError('CSV not found at {}'.format(csv_path))

    df = pd.read_csv(csv_path)
    df['Datetime (UTC)'] = pd.to_datetime(
        df['Datetime (UTC)'], format='ISO8601', utc=True)
    mask = (df['ISO3 Code'] == iso3) & (df['Datetime (UTC)'].dt.year == year)
    df = df.loc[mask, ['Datetime (UTC)', 'Price (EUR/MWhe)']]
    if len(df) == 0:
        raise ValueError(
            'No rows in {} for ISO3={} year={}'.format(csv_path, iso3, year))

    df = df.sort_values('Datetime (UTC)').set_index('Datetime (UTC)')
    prices = df['Price (EUR/MWhe)']

    full_index = pd.date_range(
        start=pd.Timestamp('{}-01-01 00:00'.format(year), tz='UTC'),
        end=pd.Timestamp('{}-12-31 23:00'.format(year), tz='UTC'),
        freq='h')
    prices = prices.reindex(full_index)

    prices = prices.clip(lower=0.0)

    assert not prices.isna().any(), (
        'NaN values in the reindexed price series for ISO3={} year={}. '
        'The raw CSV is missing hours; sanitize it before use.'.format(
            iso3, year))

    data = prices.to_numpy(dtype=float)
    meta = dict(resource='price', iso3=iso3, year=year,
                source='csv', csv_path=csv_path,
                fetched_utc=pd.Timestamp.now(tz='UTC').isoformat())
    write_cache(cache_file, data, meta)
    return TimeSeries(data, 1.0)


### Fetching normalized generation profiles from Renewables.ninja

def fetch_profile(generator_type, lat, lon, date_from, date_to, token,
                  height=None, turbine=None,
                  tilt=None, azim=None, module=None,
                  tracking=0, system_loss=0.1,
                  dataset='merra2', cache_dir=DEFAULT_CACHE_DIR):
    """Fetch a normalized production profile from renewables.ninja.

    Args:
        generator_type (str): 'wind' or 'pv'.
        lat (float): latitude of the site.
        lon (float): longitude of the site.
        date_from (str): start date, ISO format ('YYYY-MM-DD'). Inclusive.
        date_to (str): end date, ISO format ('YYYY-MM-DD'). Exclusive.
        token (str): renewables.ninja API token.
        height (float): hub height [m]. Required for type='wind'.
        turbine (str): turbine name as accepted by renewables.ninja, e.g.
            'Vestas V112 3450'. Required for type='wind'.
        tilt (float): panel tilt [degrees] from horizontal. Required for
            type='pv'.
        azim (float): panel azimuth [degrees]. 180 = south. Required for
            type='pv'.
        module (str): PV module type, e.g. 'CSi'. Used for type='pv'.
        tracking (int): 0 = fixed, 1 = single-axis, 2 = dual-axis.
            Used for type='pv'.
        system_loss (float): PV system losses as a fraction. Used for
            type='pv'.
        dataset (str): weather dataset, 'merra2' or 'sarah'.
        cache_dir (str): cache directory.

    Returns:
        TimeSeries: hourly normalized production [MW/MW], dt=1.0.

    Raises:
        ValueError: if generator_type is not 'wind' or 'pv', or if a required
            model-specific parameter is missing.
        RuntimeError: if the API returns a non-200 status.
    """
    if generator_type == 'wind':
        if height is None or turbine is None:
            raise ValueError("generator_type='wind' requires height and turbine")
        model_params = dict(height=height, turbine=turbine)
    elif generator_type == 'pv':
        if tilt is None or azim is None:
            raise ValueError("generator_type='pv' requires tilt and azim")
        model_params = dict(tilt=tilt, azim=azim, module=module,
                            tracking=tracking, system_loss=system_loss)
    else:
        raise ValueError("generator_type must be 'wind' or 'pv', got '{}'".format(generator_type))

    cache_file = cache_path(cache_dir, generator_type, lat=lat, lon=lon,
                            date_from=date_from, date_to=date_to,
                            dataset=dataset, **model_params)
    cached = read_cache(cache_file)
    if cached is not None:
        data, _ = cached
        return TimeSeries(np.array(data), 1.0)

    request_params = dict(lat=lat, lon=lon, date_from=date_from,
                          date_to=date_to, capacity=1, dataset=dataset,
                          format='json', raw='true', header='true')
    request_params.update(model_params)

    url = 'https://www.renewables.ninja/api/data/' + generator_type
    session = requests.session()
    session.headers = {'Authorization': 'Token ' + token}
    req = session.get(url, params=request_params)
    if req.status_code != 200:
        raise RuntimeError(
            'renewables.ninja request failed ({}) for type "{}": {}'.format(
                req.status_code, generator_type, req.text[:500]))
    payload = json.loads(req.text)

    df = pd.DataFrame.from_dict(payload['data'], orient='index')
    df = df.sort_index()
    data = df['electricity'].to_numpy(dtype=float)

    meta = dict(resource=generator_type, lat=lat, lon=lon,
                date_from=date_from, date_to=date_to,
                dataset=dataset, source='renewables.ninja',
                api_meta=payload.get('metadata', {}),
                fetched_utc=pd.Timestamp.now(tz='UTC').isoformat())
    meta.update(model_params)
    write_cache(cache_file, data, meta)
    return TimeSeries(data, 1.0)