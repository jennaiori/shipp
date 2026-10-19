"""
Auxiliary functions for HPP eco-design.


"""

import os
import numpy as np
import numpy_financial as npf
import pandas as pd
from shipp.timeseries import TimeSeries
from shipp.components import Production, Storage
import json
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
from pathlib import Path
import pvlib
import pyarrow
from pvlib.bifacial.infinite_sheds import get_irradiance
from pvlib.temperature import faiman
import pvlib.ivtools.sdm as sdm
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

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

def load_mdf_from_csv(csv_path, year, cache_dir=DEFAULT_CACHE_DIR):
    """Load the marginal displacement factor series for one year.

    Args:
        csv_path (str): path to the MDF CSV, columns 'datetime' and
            'MDFwind_kgCO2eperMWh'.
        year (int): calendar year to keep.
        cache_dir (str): cache directory.

    Returns:
        TimeSeries: hourly MDF [kgCO2e/MWh], dt=1.0.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if no rows match the year.
    """
    cache_file = cache_path(cache_dir, 'mdf', year=year, source=csv_path)
    cached = read_cache(cache_file)
    if cached is not None:
        data, _ = cached
        return TimeSeries(np.array(data), 1.0)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError('MDF CSV not found at {}'.format(csv_path))

    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, utc=False)
    df = df[df['datetime'].dt.year == year]
    if len(df) == 0:
        raise ValueError(
            'No MDF rows for year {} in {}'.format(year, csv_path))
    df = df.sort_values('datetime').reset_index(drop=True)
    data = df['MDFwind_kgCO2eperMWh'].to_numpy(dtype=float)

    meta = dict(resource='mdf', year=year, source=csv_path,
                fetched_utc=pd.Timestamp.now(tz='UTC').isoformat())
    write_cache(cache_file, data, meta)
    return TimeSeries(data, 1.0)

### Fetching normalized generation profiles

def get_profile(spec, sc, tx, tn):
    """Return a normalised production profile for one production asset.

    Dispatches on spec['name']. Each technology has a single source:
    wind from renewables.ninja, pv from pvlib. Wind profiles are derated
    by the 'losses' parameter of the 'wind' technology before being
    returned; pv profiles already include system losses.

    Args:
        spec (dict): production asset spec, with key 'name'.
        sc (dict): scenario dict.
        tx (dict): techs dict.
        tn (dict): technical dict.

    Returns:
        TimeSeries: normalised profile [MW/MW], dt=1.0.

    Raises:
        ValueError: if name is not 'wind' or 'pv'.
    """
    name = spec['name']

    if name == 'wind':
        perf = load_parameters(tx['params_json'], 'wind_turbine')
        ts = fetch_profile_from_rninja(
            sc['lat'], sc['lon'],
            sc['date_from'], sc['date_to'], tn['token'],
            height=perf['hub_height'], turbine=perf['name'])
        losses = perf.get('losses', 0.0)
        return TimeSeries(ts.data * (1.0 - losses), ts.dt)

    if name == 'pv':
        series = pv_generation_profile(
            sc['lat'], sc['lon'], sc['price_year'], tx['params_json'],
            tilt=tx.get('pv_tilt'), azimuth=tx.get('pv_azim'))
        return TimeSeries(series.to_numpy(dtype=float), 1.0)

    raise ValueError(
        "name must be 'wind' or 'pv', got '{}'".format(name))

def fetch_profile_from_rninja(lat, lon, date_from, date_to, token,
                              height, turbine,
                              dataset='merra2', cache_dir=DEFAULT_CACHE_DIR):
    """Fetch a normalised wind profile from renewables.ninja.

    Args:
        lat, lon (float): site coordinates.
        date_from, date_to (str): ISO date range, start inclusive, end
            exclusive.
        token (str): renewables.ninja API token.
        height (float): hub height [m].
        turbine (str): turbine model name.
        dataset (str): 'merra2' or 'sarah'.
        cache_dir (str): cache directory.

    Returns:
        TimeSeries: hourly normalised production [MW/MW], dt=1.0.

    Raises:
        RuntimeError: if the API returns a non-200 status.
    """
    model_params = dict(height=height, turbine=turbine)

    cache_file = cache_path(cache_dir, 'wind', lat=lat, lon=lon,
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

    url = 'https://www.renewables.ninja/api/data/wind'
    session = requests.session()
    session.headers = {'Authorization': 'Token ' + token}
    req = session.get(url, params=request_params)
    if req.status_code != 200:
        raise RuntimeError(
            'renewables.ninja request failed ({}): {}'.format(
                req.status_code, req.text[:500]))
    payload = json.loads(req.text)

    df = pd.DataFrame.from_dict(payload['data'], orient='index')
    df = df.sort_index()
    data = df['electricity'].to_numpy(dtype=float)

    meta = dict(resource='wind', lat=lat, lon=lon,
                date_from=date_from, date_to=date_to,
                dataset=dataset, source='renewables.ninja',
                api_meta=payload.get('metadata', {}),
                fetched_utc=pd.Timestamp.now(tz='UTC').isoformat())
    meta.update(model_params)
    write_cache(cache_file, data, meta)
    return TimeSeries(data, 1.0)


### Fetching parameters and costs

def load_parameters(params_path, technology):
    """Load performance parameters for one technology from json.

    Args:
        params_path (str): path to technology_parameters.json.
        technology (str): top-level group name, e.g. 'wind_turbine',
            'pv_module', 'lfp'.

    Returns:
        dict: {parameter: value}, without the 'name' and 'note' keys.

    Raises:
        FileNotFoundError: if the file does not exist.
        KeyError: if the group is missing.
    """
    if not os.path.isfile(params_path):
        raise FileNotFoundError(
            'Technology parameters not found at {}'.format(params_path))
    with open(params_path, 'r') as f:
        params = json.load(f)
    if technology not in params:
        raise KeyError(
            "Group '{}' missing from {}".format(technology, params_path))
    return {k: v for k, v in params[technology].items()
            if k != 'note'}


def load_costs(csv_path, technology, costs_year, inflation_csv, price_year):
    """Load capex and opex for one technology and adjust to price_year.

    Args:
        csv_path (str): path to the costs CSV.
        technology (str): technology name.
        costs_year (int): reference year of the costs in the CSV.
        inflation_csv (str): path to the inflation factor CSV.
        price_year (int): target year for adjustment.

    Returns:
        dict: {'capex': float, 'opex': float}, adjusted to price_year. Only
            keys that exist in the CSV row are present.

    Raises:
        FileNotFoundError: if either CSV does not exist.
        KeyError: if no rows match, or if a year is missing from inflation.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError('CSV not found at {}'.format(csv_path))
    df = pd.read_csv(csv_path)
    df = df[(df['technology'] == technology) & (df['year'] == costs_year)]
    if len(df) == 0:
        raise KeyError(
            'No rows for technology {} year {} in {}'.format(
                technology, costs_year, csv_path))

    if not os.path.isfile(inflation_csv):
        raise FileNotFoundError(
            'Inflation CSV not found at {}'.format(inflation_csv))
    infl = pd.read_csv(inflation_csv)
    infl['year'] = infl['year'].astype(int)
    table = dict(zip(infl['year'], infl['factor'].astype(float)))
    if costs_year not in table:
        raise KeyError(
            'Missing inflation factor for costs_year {}'.format(costs_year))
    if price_year not in table:
        raise KeyError(
            'Missing inflation factor for price_year {}'.format(price_year))
    ratio = table[price_year] / table[costs_year]

    grouped = df.groupby('parameter')['value'].mean()
    out = {}
    for param, val in grouped.items():
        if param in ('capex', 'opex'):
            out[param] = float(val) * ratio
    return out

def load_emissions(csv_path, technology):
    """Load embedded emissions for one technology from the emissions CSV.

    The year column in the CSV is ignored: embedded emissions do not vary
    by reference year. Rows with a missing value are dropped.

    Args:
        csv_path (str): path to the emissions CSV.
        technology (str): technology name.

    Returns:
        dict: {'capex': float, 'opex': float} in tonCO2e per MW or per MWh.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        KeyError: if no rows match.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError('CSV not found at {}'.format(csv_path))
    df = pd.read_csv(csv_path)
    df = df[df['technology'] == technology]
    df = df.dropna(subset=['value'])
    if len(df) == 0:
        raise KeyError(
            'No rows for technology {} in {}'.format(technology, csv_path))
    grouped = df.groupby('parameter')['value'].mean()
    return {param: float(val) for param, val in grouped.items()
            if param in ('capex', 'opex')}

### Build production and storage objects

def build_production(name, mode, params_json, costs_csv, emissions_csv,
                     inflation_csv, price_year, costs_year,
                     power_ts, n_year):
    """Return (Production, annual_cost_eco, annual_cost_env,
    capex_nominal_eco, capex_nominal_env) for a named technology.

    The Production object's p_cost is taken from the cost basis of
    `mode`. Both economic and environmental values are returned for
    reporting. annual_cost_* are per unit of installed capacity times
    n_year / lifetime. capex_nominal_* are per unit of installed
    capacity, un-annualized. Economic values are in currency adjusted to
    price_year; environmental values are in kgCO2e.

    Raises:
        ValueError: if mode is not 'economic' or 'environmental'.
    """
    if mode not in ('economic', 'environmental'):
        raise ValueError(
            "mode must be 'economic' or 'environmental', got '{}'".format(mode))

    if name == 'pv':
        components = ['pv_module', 'pv_inverter', 'pv_other']
        annual_eco = 0.0
        annual_env = 0.0
        opex_eco = 0.0
        capex_eco = 0.0
        capex_env = 0.0
        for comp in components:
            c_eco = load_costs(costs_csv, comp, costs_year,
                               inflation_csv, price_year)
            c_env = load_emissions(emissions_csv, comp)
            p = load_parameters(params_json, comp)
            annual_eco += c_eco['capex'] * n_year / p['lifetime']
            annual_env += c_env['capex'] * n_year / p['lifetime']
            opex_eco += c_eco['opex']
            capex_eco += c_eco['capex']
            capex_env += c_env['capex']
    else:
        c_eco = load_costs(costs_csv, name, costs_year,
                           inflation_csv, price_year)
        c_env = load_emissions(emissions_csv, name)
        group = 'wind_turbine' if name == 'wind' else name
        p = load_parameters(params_json, group)
        annual_eco = c_eco['capex'] * n_year / p['lifetime']
        annual_env = c_env['capex'] * n_year / p['lifetime']
        opex_eco = c_eco['opex']
        capex_eco = c_eco['capex']
        capex_env = c_env['capex']

    p_cost = annual_eco if mode == 'economic' else annual_env

    prod = Production(
        power_ts=power_ts,
        p_cost=p_cost,
        p_max=None,
        p_max_is_decision=True,
        opex_fix=opex_eco,
        opex_var=0.0,
    )
    return prod, annual_eco, annual_env, capex_eco, capex_env


def build_storage(name, mode, params_json, costs_csv, emissions_csv,
                  inflation_csv, price_year, costs_year, n_year):
    """Return (Storage, annual_cost_eco, annual_cost_env,
    capex_nominal_eco, capex_nominal_env) for a named technology.

    The Storage object's e_cost is taken from the cost basis of `mode`.
    Both economic and environmental values are returned for reporting.
    annual_cost_* are e_cost per MWh times n_year / lifetime.
    capex_nominal_* are e_cost per MWh, un-annualized. Economic values
    are in currency adjusted to price_year; environmental values are in
    kgCO2e.

    Raises:
        ValueError: if mode is not 'economic' or 'environmental'.
    """
    if mode not in ('economic', 'environmental'):
        raise ValueError(
            "mode must be 'economic' or 'environmental', got '{}'".format(mode))

    c_eco = load_costs(costs_csv, name, costs_year,
                       inflation_csv, price_year)
    c_env = load_emissions(emissions_csv, name)
    perf = load_parameters(params_json, name)

    annual_eco = c_eco['capex'] * n_year / perf['lifetime']
    annual_env = c_env['capex'] * n_year / perf['lifetime']

    e_cost = annual_eco if mode == 'economic' else annual_env

    stor = Storage(
        e_cap=None,
        p_cap=None,
        eff_in=perf['eff_charge'],
        eff_out=perf['eff_discharge'],
        p_cost=0.0,
        e_cost=e_cost,
        soc_min=1.0 - perf['dod'],
        soc_max=1.0,
        lifetime=int(perf['lifetime']),
        opex_fix=0.0,
        opex_var=0.0,
        duration=perf['duration'],
    )
    return stor, annual_eco, annual_env, c_eco['capex'], c_env['capex']

### Runner dispatch plot and summary

def plot_dispatch(os_result, price=None, grid_cap=None, title=None,
                  h_start=0, h_end=None, max_points=None,
                  start_datetime=None, price_label='Price [EUR/MWh]'):
    """Full-horizon dispatch plot with plotly.

    Args:
        os_result (OpSchedule): result from the runner.
        price (np.ndarray, optional): price series, full length.
        grid_cap (float, optional): grid cap to draw as a dashed line.
        title (str, optional): figure title.
        h_start, h_end (int): slice bounds in hours. Default full horizon.
        max_points (int, optional): downsample display to at most this many
            points. Underlying arrays are not modified.
        start_datetime (str or pd.Timestamp, optional): timestamp of hour 0
            of the underlying series. If given, the x-axis is datetime.

    Returns:
        plotly.graph_objects.Figure
    """
    n_total = len(os_result.power_out.data)
    if h_end is None:
        h_end = n_total
    sl = slice(h_start, h_end)

    dt = os_result.power_out.dt

    if start_datetime is not None:
        start = pd.Timestamp(start_datetime)
        x_full = pd.date_range(start=start + pd.Timedelta(hours=h_start),
                               periods=h_end - h_start, freq=f'{int(dt * 60)}min')
    else:
        x_full = np.arange(h_start, h_end) * dt

    prod = [p.data[sl] for p in os_result.production_p]
    stor = [s.data[sl] for s in os_result.storage_p]
    p_cur = os_result.p_curtail.data[sl]

    total_avail = sum(prod)
    frac = np.zeros_like(total_avail)
    mask = total_avail > 0
    frac[mask] = p_cur[mask] / total_avail[mask]
    prod_net = [p * (1 - frac) for p in prod]

    x = x_full
    if max_points is not None and len(x_full) > max_points:
        step = int(np.ceil(len(x_full) / max_points))
        x = x_full[::step]
        prod_net = [p[::step] for p in prod_net]
        stor = [s[::step] for s in stor]

    fig = go.Figure()

    for i, p in enumerate(prod_net):
        fig.add_trace(go.Bar(
            x=x, y=p, name=f'production {i+1}',
            marker_color=['#1f77b4', '#17becf'][i % 2],
        ))

    if len(stor) >= 1:
        fig.add_trace(go.Bar(
            x=x, y=np.maximum(stor[0], 0), name='storage discharge',
            marker_color='#ff7f0e',
        ))
        fig.add_trace(go.Bar(
            x=x, y=np.minimum(stor[0], 0), name='storage charge',
            marker_color='#d62728',
        ))

    if grid_cap is not None:
        fig.add_hline(y=grid_cap, line_dash='dash', line_color='black',
                      annotation_text='grid cap',
                      annotation_position='top left')

    fig.update_layout(
        barmode='relative',
        xaxis_title='Time (UTC)',
        yaxis_title='Power [MW]',
        title=title or '',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        height=500,
    )

    if price is not None:
        price_arr = np.asarray(price)[sl]
        price_x = x_full
        if max_points is not None and len(price_arr) > max_points:
            step = int(np.ceil(len(price_arr) / max_points))
            price_arr = price_arr[::step]
            price_x = x_full[::step]
        fig.add_trace(go.Scatter(
            x=price_x, y=price_arr, name=price_label,
            yaxis='y2', mode='lines', line=dict(color='grey', width=1),
        ))
        fig.update_layout(
            yaxis2=dict(title=price_label, overlaying='y',
                        side='right', showgrid=False),
        )

    if start_datetime is not None:
        window_hours = h_end - h_start
        if window_hours <= 24 * 3:
            tick_fmt = '%b %d %H:%M'
            tick_ms = 3600 * 1000
            tick_angle = -45
        elif window_hours <= 24 * 14:
            tick_fmt = '%b %d %H:%M'
            tick_ms = 6 * 3600 * 1000   # every 6 hours
            tick_angle = -45
        else:
            tick_fmt = '%b %d'
            tick_ms = 24 * 3600 * 1000
            tick_angle = -30

        fig.update_xaxes(
            tickformat=tick_fmt,
            dtick=tick_ms,
            tickangle=tick_angle,
        )

    fig.show()
    return fig

def build_summary(mode, tn, os_result, prod1_ts, prod2_ts,
                  price_eco, price_env,
                  annual_cost_eco, annual_cost_env,
                  opex_eco,
                  capex_nominal_eco, capex_nominal_env,
                  discount_rate_eco, discount_rate_env, n_year,
                  t_wall, caught):
    """Assemble the ex-post summary for a sizing case.

    Args:
        mode (str): 'economic' or 'environmental'.
        tn, (dict): technical dict.
        os_result (OpSchedule): kernel output.
        prod1_ts, prod2_ts (TimeSeries): derated profiles used in the solve.
        price_eco (TimeSeries or None): day-ahead price series, dt=1.0.
        price_env (TimeSeries or None): MDF series, dt=1.0.
        annual_cost_eco (dict): {'prod1','prod2','stor1','stor2'}
            annualized monetary cost per unit (already times n_year/lifetime).
        annual_cost_env (dict): same shape, in kgCO2e.
        opex_eco (dict): {'prod1','prod2'} fixed opex per MW per year.
        capex_nominal_eco (dict): un-annualized monetary capex per unit.
        capex_nominal_env (dict): un-annualized kgCO2e capex per unit.
        discount_rate_eco (float): discount rate for the economic NPV.
        discount_rate_env (float): discount rate for the environmental NPV.
        n_year (int): horizon.
        t_wall (float): wall time in seconds.
        caught (list): warnings captured during the solve.

    Returns:
        dict: summary.
    """
    x_prod1 = os_result.production_list[0].p_max
    x_prod2 = os_result.production_list[1].p_max
    e_stor1 = os_result.storage_list[0].e_cap
    e_stor2 = os_result.storage_list[1].e_cap

    dt = os_result.power_out.dt
    power_out = os_result.power_out.data

    raw = prod1_ts.data * x_prod1 + prod2_ts.data * x_prod2
    delivered = power_out.sum()
    curtailed = os_result.p_curtail.data.sum()
    charge1 = -np.minimum(os_result.storage_p[0].data, 0).sum()
    discharge1 = np.maximum(os_result.storage_p[0].data, 0).sum()
    charge2 = -np.minimum(os_result.storage_p[1].data, 0).sum()
    discharge2 = np.maximum(os_result.storage_p[1].data, 0).sum()
    charge_total = charge1 + charge2
    discharge_total = discharge1 + discharge2

    # --- economic NPV ---
    capex_eff_eco = (
        annual_cost_eco['prod1'] * x_prod1
        + annual_cost_eco['prod2'] * x_prod2
        + annual_cost_eco['stor1'] * e_stor1
        + annual_cost_eco['stor2'] * e_stor2
    )
    factor_eco = npf.npv(discount_rate_eco, np.ones(n_year)) - 1
    if price_eco is not None:
        revenue_eco = float(np.dot(price_eco.data, power_out)) * dt
        opex_total_eco = (opex_eco['prod1'] * x_prod1
                          + opex_eco['prod2'] * x_prod2)
        npv_eco = (factor_eco * revenue_eco
                   - factor_eco * opex_total_eco
                   - capex_eff_eco)
    else:
        revenue_eco = None
        opex_total_eco = None
        npv_eco = None

    # --- environmental NPV ---
    capex_eff_env = (
        annual_cost_env['prod1'] * x_prod1
        + annual_cost_env['prod2'] * x_prod2
        + annual_cost_env['stor1'] * e_stor1
        + annual_cost_env['stor2'] * e_stor2
    )
    factor_env = npf.npv(discount_rate_env, np.ones(n_year)) - 1
    if price_env is not None:
        displaced_env = float(np.dot(price_env.data, power_out)) * dt
        npv_env = factor_env * displaced_env - capex_eff_env
    else:
        displaced_env = None
        npv_env = None

    capex_nominal_eco_total = (
        capex_nominal_eco['prod1'] * x_prod1
        + capex_nominal_eco['prod2'] * x_prod2
        + capex_nominal_eco['stor1'] * e_stor1
        + capex_nominal_eco['stor2'] * e_stor2
    )
    capex_nominal_env_total = (
        capex_nominal_env['prod1'] * x_prod1
        + capex_nominal_env['prod2'] * x_prod2
        + capex_nominal_env['stor1'] * e_stor1
        + capex_nominal_env['stor2'] * e_stor2
    )

    return {
        'mode': mode,
        'formulation': tn.get('formulation', 'milp'),
        'solver': tn.get('name_solver', 'gurobi'),
        'wall_time_s': t_wall,
        'solver_time_s': getattr(os_result, 'time', None),
        'warnings': [str(w.message) for w in caught],

        # Configuration
        'x_prod1_mw': x_prod1,
        'x_prod2_mw': x_prod2,
        'p_stor1_mw': os_result.storage_list[0].p_cap,
        'e_stor1_mwh': e_stor1,
        'p_stor2_mw': os_result.storage_list[1].p_cap,
        'e_stor2_mwh': e_stor2,

        # Dispatch
        'curtail_frac': curtailed / raw.sum() if raw.sum() > 0 else 0.0,
        'rte_realized': (discharge_total / charge_total
                         if charge_total > 0 else None),

        # Economic dimension
        'nominal_capex_meur': capex_nominal_eco_total * 1e-6,
        'npv_meur': (npv_eco * 1e-6
                               if npv_eco is not None else None),
        'npv_per_capex_eco': (npv_eco / capex_nominal_eco_total
                              if (npv_eco is not None
                                  and capex_nominal_eco_total > 0)
                              else None),

        # Environmental dimension
        'embedded_emissions_mkgco2e': capex_nominal_env_total * 1e-6,
        'net_emissions_mkgco2e': (npv_env * 1e-6
                              if npv_env is not None else None),
        'npv_per_capex_env': (npv_env / capex_nominal_env_total
                              if (npv_env is not None
                                  and capex_nominal_env_total > 0)
                              else None),

        # Checks
        'loss_check_ok': os_result.check_losses(1e-4),
    }

### Solar PV modeling with pvlib

def get_pvgis_hourly(lat, lon, year, cache_dir="pvgis_cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{lat:.4f}_{lon:.4f}_{year}_horizontal"
    path = cache_dir / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    data, meta = pvlib.iotools.get_pvgis_hourly(
        latitude=lat, longitude=lon,
        start=year, end=year,
        raddatabase="PVGIS-SARAH3",
        components=False,
        surface_tilt=0,
        surface_azimuth=0,
        map_variables=True,
    )
    data.to_parquet(path)
    return data


def pv_generation_profile(lat, lon, year, params_path, tilt=None, azimuth=None,
                          cache_dir=DEFAULT_CACHE_DIR,
                          pvgis_cache_dir="pvgis_cache",
                          dc_model="cec"):
    """Return a normalized PV AC generation profile from pvlib.

    Args:
        lat, lon (float): site coordinates.
        year (int): calendar year of the PVGIS weather data.
        params_path (str): path to technology_parameters.json. Reads the
            'pv_module', 'pv_inverter', and 'pv_farm' groups.
        tilt (float, optional): fixed array tilt [deg]. If None, or if
            azimuth is also None, tilt and azimuth are chosen by an annual
            POA sweep.
        azimuth (float, optional): fixed array azimuth [deg].
        cache_dir (str): PVGIS cache directory.
        dc_model (str): 'cec' or 'pvwatts'.

    Returns:
        pd.Series: hourly AC output per MW of inverter AC nameplate
            [MW/MWac], dt=1 h, indexed by UTC timestamp.

    Raises:
        FileNotFoundError: if params_path does not exist.
        KeyError: if a required group or field is missing from the JSON.
        ValueError: if dc_model is not 'cec' or 'pvwatts', or if
            n_inputs_per_mppt does not match n_mppt.
    """

    ### search for data in the cache 
    cache_file = cache_path(cache_dir, 'pv',
                        lat=round(float(lat), 4), lon=round(float(lon), 4),
                        year=int(year), tilt=tilt, azimuth=azimuth,
                        dc_model=dc_model)
    cached = read_cache(cache_file)
    if cached is not None:
        data, _ = cached
        return pd.Series(np.array(data))


    ### parameter loading
    if not os.path.isfile(params_path):
        raise FileNotFoundError(
            'Technology parameters not found at {}'.format(params_path))
    with open(params_path, 'r') as f:
        params = json.load(f)    

    module   = params['pv_module']
    inverter = params['pv_inverter']
    farm     = params['pv_farm']

    albedo = farm['albedo']
    gcr    = farm['gcr']
    height = farm['height']
    ilr    = farm['ilr']
    u0     = farm['u0']
    u1     = farm['u1']
    pitch  = module['length_m'] / gcr

    ### pvgis hourly data and irradiance component calculations 
    data = get_pvgis_hourly(lat, lon, year, cache_dir=pvgis_cache_dir)
    ghi = data["poa_global"].clip(lower=0).rename("ghi")

    solpos = pvlib.solarposition.get_solarposition(data.index, lat, lon)
    solar_zenith = solpos["apparent_zenith"]
    solar_azimuth = solpos["azimuth"]

    erbs = pvlib.irradiance.erbs(
        ghi=ghi,
        zenith=solar_zenith,
        datetime_or_doy=data.index,
    )
    dni = erbs["dni"].clip(lower=0).fillna(0).rename("dni")
    dhi = erbs["dhi"].clip(lower=0).fillna(0).rename("dhi")

    weather = pd.DataFrame({
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "temp_air": data["temp_air"],
        "wind_speed": data["wind_speed"],
    })
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)

    ### optimal tilt and azimuth for a bifacial module
    if tilt is not None and azimuth is not None:
        best_tilt = tilt
        best_az = azimuth
    else:
        tilts = np.arange(15, 55, 1)
        azimuths = np.arange(165, 195, 1)
        rows = []
        for tlt in tilts:
            for az in azimuths:
                poa = get_irradiance(
                        surface_tilt=tlt,
                        surface_azimuth=az,
                        solar_zenith=solar_zenith,
                        solar_azimuth=solar_azimuth,
                        gcr=gcr,
                        height=height,
                        pitch=pitch,
                        ghi=weather["ghi"],
                        dhi=weather["dhi"],
                        dni=weather["dni"],
                        albedo=albedo,
                        dni_extra=dni_extra,
                        bifaciality=module["bifaciality"],
                        model="perez",
                    )
                annual = poa["poa_global"].sum() / 1000.0
                rows.append({"tilt": tlt, "azimuth": az, "annual_kwh_m2": annual})
        sweep_bifacial = pd.DataFrame(rows)
        best_bifacial = sweep_bifacial.loc[sweep_bifacial["annual_kwh_m2"].idxmax()]
        best_tilt = float(best_bifacial["tilt"])
        best_az = float(best_bifacial["azimuth"])

    ### recalculate yearly irradiance time-series for given or optimized tilt and azimuth
    poa_best = get_irradiance(
        surface_tilt=best_tilt,
        surface_azimuth=best_az,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        gcr=gcr,
        height=height,
        pitch=pitch,
        ghi=weather["ghi"],
        dhi=weather["dhi"],
        dni=weather["dni"],
        albedo=albedo,
        dni_extra=dni_extra,
        bifaciality=[module["bifaciality"]],
        model="perez",
    )

    effective_irradiance = poa_best["poa_global"].clip(lower=0)

    ### string topology calculation
    T_STC = 25.0
    T_cold = weather["temp_air"].min()   # degC, from PVGIS weather

    cell_temp = faiman(
        poa_global=effective_irradiance,
        temp_air=weather["temp_air"],
        wind_speed=weather["wind_speed"],
        u0=u0,
        u1=u1,
    )

    T_hot_cell = float(cell_temp.max()) # degC, using Faiman model


    # number of modules
    p_dc_target = inverter["pac0"] * ilr
    n_modules_total = int(round(p_dc_target / module["pdc0"]))

    # max string length constrained by inverter max voltage at cold temp
    v_oc_cold = module["v_oc"] * (1 + module["beta_voc"] * (T_cold - T_STC))
    n_s_max = int(inverter["v_dc_max"] // v_oc_cold)

    # min string length constrained by mppt lower voltage bound at hot temp
    v_mp_hot = module["v_mp"] * (1 + module["beta_voc"] * (T_hot_cell - T_STC))
    n_s_min = int(-(-inverter["v_mppt_min"] // v_mp_hot))   # ceil division

    n_s = n_s_max  # maximize string length to reduce n_p and balance mppt inputs

    # sweep parallel strings and pick the ILR closest to target
    max_strings_per_mppt_by_imp = int(inverter["i_mppt_max"] // module["i_mp"])
    max_strings_per_mppt_by_isc = int(inverter["i_sc_mppt_max"] // module["i_sc"])
    max_strings_per_mppt = min(max_strings_per_mppt_by_imp, max_strings_per_mppt_by_isc)
    max_strings_per_mppt = min(max_strings_per_mppt, min(inverter["n_inputs_per_mppt"]))

    p_string = n_s * module["pdc0"]  # DC power per string

    best = None  # (abs_err, n_p, ilr_actual)
    for n_p_cand in range(1, max_strings_per_mppt * inverter["n_mppt"] + 1):
        ilr_cand = n_p_cand * p_string / inverter["pac0"]
        err = abs(ilr_cand - ilr)
        if best is None or err < best[0]:
            best = (err, n_p_cand, ilr_cand)

    _, n_p, ilr_actual = best

    # distribute n_p across MPPTs, cap respected after remainder
    strings_per_mppt = [min(max_strings_per_mppt, n_p // inverter["n_mppt"])] * inverter["n_mppt"]
    remaining = n_p - sum(strings_per_mppt)
    for i in range(remaining):
        strings_per_mppt[i] += 1

    ### shared parameters 
    inverter_parameters = {
        "pdc0": inverter["pac0"] / inverter["eta_inv_nom"],  # based on pvwatts assumption
        "eta_inv_nom": inverter["eta_inv_nom"],
    }

    temperature_model_parameters = {"u0": u0, "u1": u1}
    location = Location(lat, lon, tz="UTC")

    ### pvsystem and modelchain setup
    if dc_model == "cec":
        cec_params = sdm.fit_cec_sam(
            celltype="monoSi",
            v_mp=module["v_mp"],
            i_mp=module["i_mp"],
            v_oc=module["v_oc"],
            i_sc=module["i_sc"],
            alpha_sc=module["alpha_isc"] * module["i_sc"],
            beta_voc=module["beta_voc"] * module["v_oc"],
            gamma_pmp=module["gamma_pdc"]*100,
            cells_in_series=module["cells_in_series"],
            temp_ref=25,
        )
        I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = cec_params

        module_parameters = {
            "alpha_sc": module["alpha_isc"] * module["i_sc"],        
            "I_L_ref": I_L_ref,
            "I_o_ref": I_o_ref,
            "R_sh_ref": R_sh_ref,
            "R_s": R_s,
            "Adjust": Adjust,
            "a_ref": a_ref,
            "N": module["cells_in_series"],
        }

        system = PVSystem(
            surface_tilt=best_tilt,
            surface_azimuth=best_az,
            module_parameters=module_parameters,
            inverter_parameters=inverter_parameters,
            temperature_model_parameters=temperature_model_parameters,
            modules_per_string=n_s,
            strings_per_inverter=n_p,
        )

        mc = ModelChain(
            system, location,
            aoi_model="no_loss",
            spectral_model="no_loss",
            dc_model="cec",
            ac_model="pvwatts",
            temperature_model="faiman",
            losses_model="pvwatts",
            transposition_model="perez",
        )

    elif dc_model == "pvwatts":
        module_parameters = {
            "pdc0": module["pdc0"],
            "gamma_pdc": module["gamma_pdc"],
        }

        system = PVSystem(
            surface_tilt=best_tilt,
            surface_azimuth=best_az,
            module_parameters=module_parameters,
            inverter_parameters=inverter_parameters,
            temperature_model_parameters=temperature_model_parameters,
            modules_per_string=n_s,
            strings_per_inverter=n_p,
        )

        location = Location(lat, lon, tz="UTC")

        mc = ModelChain(
            system, location,
            aoi_model="no_loss",
            spectral_model="no_loss",
            dc_model="pvwatts",
            ac_model="pvwatts",
            temperature_model="faiman",
            losses_model="pvwatts",
            transposition_model="perez",
        )

    else:
        raise ValueError(f"dc_model must be 'cec' or 'pvwatts', got '{dc_model}'")

    ### run the model and return AC generation profile in MW per MW of inverter capacity
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='invalid value encountered in divide',
            category=RuntimeWarning)
        mc.run_model(weather)
    ac = mc.results.ac["p_mp"].clip(lower=0)
    ac_per_mwac = ac / inverter["pac0"]

    meta = dict(resource='pv', lat=lat, lon=lon, year=year,
                tilt=best_tilt, azimuth=best_az,
                dc_model=dc_model, source='pvlib',
                fetched_utc=pd.Timestamp.now(tz='UTC').isoformat())
    write_cache(cache_file, ac_per_mwac.to_numpy(dtype=float), meta)
    return ac_per_mwac