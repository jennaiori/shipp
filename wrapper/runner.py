# runner.py

import time
import warnings
import pandas as pd

from auxiliary_functions import (
    load_price_from_csv,
    load_mdf_from_csv,
    build_production,
    build_storage,
    get_profile,
    BETA_OBJ_DEFAULT,
    plot_dispatch,
    build_summary,
)
from shipp.components import Storage
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo_sizing


def run_case(case):
    """Run a single sizing/dispatch case.

    mode='economic' uses the day-ahead price as the price series and
    monetary capex/opex for the assets. mode='environmental' uses the
    marginal displacement factor series as the price and embedded
    emissions as the asset costs.

    Args:
        case (dict): nested case description with 'scenario', 'techs',
            and 'technical' levels.

    Returns:
        (os_result, summary)
    """
    sc = case['scenario']
    tx = case['techs']
    tn = case['technical']

    mode = tn['mode']
    if mode not in ('economic', 'environmental'):
        raise ValueError(
            "mode must be 'economic' or 'environmental', got '{}'".format(mode))

    n = sc['n']
    dt = sc.get('dt', 1.0)
    n_year = sc['n_year']
    price_year = sc['price_year']

    p1_spec = tx['prod1']
    p2_spec = tx['prod2']
    s1_spec = tx['stor1']
    s2_spec = tx['stor2']

    # --- profiles ---
    prod1_ts = get_profile(p1_spec, sc, tx, tn)
    prod2_ts = get_profile(p2_spec, sc, tx, tn)

    price_eco = load_price_from_csv(
        sc['price_iso3'], price_year, sc['price_csv'])
    price_env = load_mdf_from_csv(tx['mdf_csv'], price_year)

    prod1_ts = TimeSeries(prod1_ts.data[:n], dt)
    prod2_ts = TimeSeries(prod2_ts.data[:n], dt)
    price_eco = TimeSeries(price_eco.data[:n], dt)
    price_env = TimeSeries(price_env.data[:n], dt)

    price_ts = price_eco if mode == 'economic' else price_env

    # --- assets ---
    (prod1, annual_eco_p1, annual_env_p1,
     capex_eco_p1, capex_env_p1) = build_production(
        p1_spec['name'], tx['params_csv'], tx['costs_csv'],
        tx['emissions_csv'], tx['inflation_csv'], price_year,
        p1_spec['costs_year'], prod1_ts, n_year)
    (prod2, annual_eco_p2, annual_env_p2,
     capex_eco_p2, capex_env_p2) = build_production(
        p2_spec['name'], tx['params_csv'], tx['costs_csv'],
        tx['emissions_csv'], tx['inflation_csv'], price_year,
        p2_spec['costs_year'], prod2_ts, n_year)

    if s1_spec['name'] is None:
        stor1 = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                        p_cost=0.0, e_cost=0.0, duration=1.0)
        annual_eco_s1 = 0.0
        annual_env_s1 = 0.0
        capex_eco_s1 = 0.0
        capex_env_s1 = 0.0
    else:
        (stor1, annual_eco_s1, annual_env_s1,
         capex_eco_s1, capex_env_s1) = build_storage(
            s1_spec['name'], tx['params_csv'], tx['costs_csv'],
            tx['emissions_csv'], tx['inflation_csv'], price_year,
            s1_spec['costs_year'], n_year)

    if s2_spec['name'] is None:
        stor2 = Storage(e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
                        p_cost=0.0, e_cost=0.0, duration=1.0)
        annual_eco_s2 = 0.0
        annual_env_s2 = 0.0
        capex_eco_s2 = 0.0
        capex_env_s2 = 0.0
    else:
        (stor2, annual_eco_s2, annual_env_s2,
         capex_eco_s2, capex_env_s2) = build_storage(
            s2_spec['name'], tx['params_csv'], tx['costs_csv'],
            tx['emissions_csv'], tx['inflation_csv'], price_year,
            s2_spec['costs_year'], n_year)

    if mode == 'environmental':
        prod1.p_cost = annual_env_p1
        prod2.p_cost = annual_env_p2
        if s1_spec['name'] is not None:
            stor1.e_cost = annual_env_s1
        if s2_spec['name'] is not None:
            stor2.e_cost = annual_env_s2

    annual_cost_eco = {
        'prod1': annual_eco_p1, 'prod2': annual_eco_p2,
        'stor1': annual_eco_s1, 'stor2': annual_eco_s2,
    }
    annual_cost_env = {
        'prod1': annual_env_p1, 'prod2': annual_env_p2,
        'stor1': annual_env_s1, 'stor2': annual_env_s2,
    }
    capex_nominal_eco = {
        'prod1': capex_eco_p1, 'prod2': capex_eco_p2,
        'stor1': capex_eco_s1, 'stor2': capex_eco_s2,
    }
    capex_nominal_env = {
        'prod1': capex_env_p1, 'prod2': capex_env_p2,
        'stor1': capex_env_s1, 'stor2': capex_env_s2,
    }
    opex_eco = {'prod1': prod1.opex_fix, 'prod2': prod2.opex_fix}

    # --- budget (economic mode only) ---
    budget = sc.get('budget')
    if budget is not None and mode == 'economic':
        tol = tn.get('budget_tol', 1e-4)
        i_max = budget * (1 + tol)
        i_min = budget * (1 - tol)
        nominal_costs = {
            'prod1': capex_eco_p1,
            'prod2': capex_eco_p2,
            'stor1_p': 0.0,
            'stor1_e': capex_eco_s1,
            'stor2_p': 0.0,
            'stor2_e': capex_eco_s2,
        }
    else:
        i_max = None
        i_min = None
        nominal_costs = None

    # --- solve ---
    discount_rate = 0.0 if mode == 'environmental' else sc['discount_rate']

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        os_result = solve_lp_pyomo_sizing(
            price_ts=price_ts,
            prod1=prod1, prod2=prod2,
            prof1_unit=prod1_ts, prof2_unit=prod2_ts,
            stor1=stor1, stor2=stor2,
            discount_rate=discount_rate, n_year=n_year,
            p_grid_max=sc['p_grid_max'], n=n,
            i_max=i_max, i_min=i_min,
            nominal_costs=nominal_costs,
            beta_obj=tn.get('beta_obj', BETA_OBJ_DEFAULT),
            options=dict(
                name_solver=tn.get('name_solver', 'gurobi'),
                formulation=tn.get('formulation', 'milp')),
        )
    t_wall = time.perf_counter() - t0

    if not tn.get('summarize', True):
        return os_result, None

    summary = build_summary(
        mode=mode, tn=tn, sc=sc,
        p1_spec=p1_spec, p2_spec=p2_spec,
        s1_spec=s1_spec, s2_spec=s2_spec,
        os_result=os_result,
        prod1_ts=prod1_ts, prod2_ts=prod2_ts,
        price_eco=price_eco, price_env=price_env,
        annual_cost_eco=annual_cost_eco,
        annual_cost_env=annual_cost_env,
        opex_eco=opex_eco,
        capex_nominal_eco=capex_nominal_eco,
        capex_nominal_env=capex_nominal_env,
        discount_rate_eco=sc['discount_rate'],
        discount_rate_env=0.0,
        n_year=n_year,
        t_wall=t_wall, caught=caught,
    )

    if tn.get('plot_dispatch', False):
        start_dt = pd.Timestamp(
            f"{price_year}-01-01 00:00", tz='UTC').tz_localize(None)
        price_label = ('Price [EUR/MWh]' if mode == 'economic'
                       else 'MDF [kgCO2e/MWh]')
        plot_dispatch(
            os_result,
            price=price_ts.data,
            grid_cap=sc['p_grid_max'],
            h_start=0, h_end=24 * 7,
            title=f"{mode.capitalize()} dispatch",
            max_points=2000,
            start_datetime=start_dt,
            price_label=price_label,
        )

    return os_result, summary


if __name__ == '__main__':
    case = {
        'scenario': {
            'lat': 52.3, 'lon': 5.3,
            'date_from': '2023-01-01', 'date_to': '2024-01-01',
            'price_iso3': 'NLD', 'price_year': 2023,
            'price_csv': 'ember_prices.csv',
            'p_grid_max': 300,
            'budget': None,
            'n': 24 * 365,
            'n_year': 25,
            'discount_rate': 0.07,
        },
        'techs': {
            'params_csv': 'technology_parameters.csv',
            'costs_csv': 'technology_costs.csv',
            'emissions_csv': 'technology_emissions.csv',
            'inflation_csv': 'inflation.csv',
            'mdf_csv': 'mdf_nl.csv',
            'wind_height': 90, 'wind_turbine': 'Vestas V150 4500',
            'pv_tilt': 35, 'pv_azim': 180,
            'prod1': {'name': 'wind', 'source': 'rninja', 'costs_year': 2020},
            'prod2': {'name': 'pv',   'source': 'rninja', 'costs_year': 2025},
            'stor1': {'name': 'lfp',  'costs_year': 2020},
            'stor2': {'name': None,   'costs_year': 2020},
        },
        'technical': {
            'mode': 'environmental',
            'token': '96e53647b34ef4fb37b6958e9285eff9938ad7ce',
            'budget_tol': 1e-4,
            'formulation': 'milp',
            'name_solver': 'gurobi',
            'beta_obj': BETA_OBJ_DEFAULT,
            'summarize': True,
            'plot_dispatch': True,
        },
    }
    os_result, summary = run_case(case)
    if summary is not None:
        for k, v in summary.items():
            print(f"{k:>24}: {v}")