# runner.py

import time
import warnings
import pandas as pd
import os

from auxiliary_functions import (
    load_price_from_csv,
    load_mdf_from_csv,
    build_production,
    build_storage,
    get_profile,
    BETA_OBJ_DEFAULT,
    plot_dispatch,
    plot_pareto,
    build_summary,
)
from shipp.components import Storage
from shipp.timeseries import TimeSeries
from shipp.kernel_sizing import (
    build_sizing_model,
    solve_sizing_SOO,
    solve_sizing_MOO,
)


def run_case(case):
    """Run a sizing/dispatch case with the configured method.

    Args:
        case (dict): nested case description with 'scenario', 'techs',
            and 'technical' levels. ``technical['method']`` selects
            'SOO' or 'weighted_sum' (default 'SOO').

    Returns:
        (sols, summaries). ``sols`` is the list of solution dicts from
        the kernel; ``summaries`` is the matching list from
        ``build_summary``.
    """
    sc = case['scenario']
    tx = case['techs']
    tn = case['technical']
    verbose = tn.get('verbose', True)

    mode = tn['mode']
    if mode not in ('economic', 'environmental'):
        raise ValueError(
            "mode must be 'economic' or 'environmental', got '{}'".format(
                mode))
    method = tn.get('method', 'SOO')
    if method not in ('SOO', 'weighted_sum', 'augmecon'):
        raise ValueError(
            "method must be 'SOO', 'weighted_sum' or 'augmecon', got "
            "'{}'".format(method))

    n = sc['n']
    dt = sc.get('dt', 1.0)
    n_year = sc['n_year']
    price_year = sc['price_year']

    p1_spec = tx['prod1']
    p2_spec = tx['prod2']
    s1_spec = tx['stor1']
    s2_spec = tx['stor2']

    if verbose:
        print("Generating profiles...", end=" ", flush=True)
    t = time.perf_counter()
    prod1_ts = get_profile(p1_spec, sc, tx, tn)
    prod2_ts = get_profile(p2_spec, sc, tx, tn)
    if verbose:
        print("{:.2f} s".format(time.perf_counter() - t))

    if verbose:
        print("Loading prices...", end=" ", flush=True)
    t = time.perf_counter()
    price_eco = load_price_from_csv(
        sc['price_iso3'], price_year, sc['price_csv'])
    price_env = load_mdf_from_csv(tx['mdf_csv'], price_year)
    prod1_ts = TimeSeries(prod1_ts.data[:n], dt)
    prod2_ts = TimeSeries(prod2_ts.data[:n], dt)
    price_eco = TimeSeries(price_eco.data[:n], dt)
    price_env = TimeSeries(price_env.data[:n], dt)
    if verbose:
        print("{:.2f} s".format(time.perf_counter() - t))

    if verbose:
        print("Building assets...", end=" ", flush=True)
    t = time.perf_counter()
    (prod1, annual_eco_p1, annual_env_p1,
     capex_eco_p1, capex_env_p1) = build_production(
        p1_spec['name'], tn['mode'], tx['params_json'], tx['costs_csv'],
        tx['emissions_csv'], tx['inflation_csv'], price_year,
        p1_spec['costs_year'], prod1_ts, n_year)
    (prod2, annual_eco_p2, annual_env_p2,
     capex_eco_p2, capex_env_p2) = build_production(
        p2_spec['name'], tn['mode'], tx['params_json'], tx['costs_csv'],
        tx['emissions_csv'], tx['inflation_csv'], price_year,
        p2_spec['costs_year'], prod2_ts, n_year)

    if s1_spec['name'] is None:
        stor1 = Storage(
            e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
            p_cost_economic=0.0, p_cost_environmental=0.0,
            e_cost_economic=0.0, e_cost_environmental=0.0,
            opex_fix_economic=0.0, opex_fix_environmental=0.0,
            opex_var_economic=0.0, opex_var_environmental=0.0,
            duration=1.0)
        annual_eco_s1 = 0.0
        annual_env_s1 = 0.0
        capex_eco_s1 = 0.0
        capex_env_s1 = 0.0
    else:
        (stor1, annual_eco_s1, annual_env_s1,
         capex_eco_s1, capex_env_s1) = build_storage(
            s1_spec['name'], tn['mode'], tx['params_json'], tx['costs_csv'],
            tx['emissions_csv'], tx['inflation_csv'], price_year,
            s1_spec['costs_year'], n_year)

    if s2_spec['name'] is None:
        stor2 = Storage(
            e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
            p_cost_economic=0.0, p_cost_environmental=0.0,
            e_cost_economic=0.0, e_cost_environmental=0.0,
            opex_fix_economic=0.0, opex_fix_environmental=0.0,
            opex_var_economic=0.0, opex_var_environmental=0.0,
            duration=1.0)
        annual_eco_s2 = 0.0
        annual_env_s2 = 0.0
        capex_eco_s2 = 0.0
        capex_env_s2 = 0.0
    else:
        (stor2, annual_eco_s2, annual_env_s2,
         capex_eco_s2, capex_env_s2) = build_storage(
            s2_spec['name'], tn['mode'], tx['params_json'], tx['costs_csv'],
            tx['emissions_csv'], tx['inflation_csv'], price_year,
            s2_spec['costs_year'], n_year)

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
    opex_eco = {'prod1': prod1.opex_fix_economic,
                'prod2': prod2.opex_fix_economic}

    budget = sc.get('budget')
    if budget is not None:
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

    if verbose:
        print("{:.2f} s".format(time.perf_counter() - t))

    if verbose:
        print("Building model...", end=" ", flush=True)
    t = time.perf_counter()
    model, meta = build_sizing_model(
        price_eco=price_eco,
        price_env=price_env,
        prod1=prod1, prod2=prod2,
        prof1_unit=prod1_ts, prof2_unit=prod2_ts,
        stor1=stor1, stor2=stor2,
        discount_rate_eco=sc['discount_rate'],
        discount_rate_env=0.0,
        n_year=n_year,
        p_grid_max=sc['p_grid_max'], n=n,
        i_max=i_max, i_min=i_min,
        nominal_costs=nominal_costs,
        beta_obj=tn.get('beta_obj', BETA_OBJ_DEFAULT),
        options=dict(
            name_solver=tn.get('name_solver', 'gurobi'),
            formulation=tn.get('formulation', 'milp')),
    )
    if verbose:
        print("{:.2f} s".format(time.perf_counter() - t))

    if verbose:
        print("Running {}...".format(method), end=" ", flush=True)
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == 'SOO':
            sol = solve_sizing_SOO(model, meta, dimension=mode)
            sols = [sol]
        else:
            sols = solve_sizing_MOO(
                model, meta, method=method,
                weights=tn.get('weight_sweep'),
                n_points=tn.get('n_points', 11),
                verbose=verbose)
    t_wall = time.perf_counter() - t
    if verbose:
        print("{:.2f} s".format(t_wall))

    summaries = []
    for sol in sols:
        summary = build_summary(
            mode=mode, tn=tn,
            os_result=sol['os_result'],
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
        summary['method'] = sol['method']
        summary['dimension'] = sol.get('dimension', mode)
        if 'weight' in sol:
            summary['weight'] = sol['weight']
        summary['objectives'] = sol['objectives']
        summaries.append(summary)

    if tn.get('plot_dispatch', False) and method == 'SOO':
        start_dt = pd.Timestamp(
            "{}-01-01 00:00".format(price_year), tz='UTC').tz_localize(None)
        price_label = ('Price [EUR/MWh]' if mode == 'economic'
                       else 'MDF [kgCO2e/MWh]')
        price_series = price_eco if mode == 'economic' else price_env
        plot_dispatch(
            sols[0]['os_result'],
            price=price_series.data,
            grid_cap=sc['p_grid_max'],
            h_start=0, h_end=24 * 365,
            title="{} SOO dispatch".format(mode.capitalize()),
            max_points=8760,
            start_datetime=start_dt,
            price_label=price_label,
        )

    return sols, summaries


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
            'params_json': 'technology_parameters.json',
            'costs_csv': 'technology_costs.csv',
            'emissions_csv': 'technology_emissions.csv',
            'inflation_csv': 'inflation.csv',
            'mdf_csv': 'mdf_nl.csv',
            'prod1': {'name': 'wind', 'costs_year': 2020},
            'prod2': {'name': 'pv',   'costs_year': 2025},
            'stor1': {'name': 'lfp',  'costs_year': 2020},
            'stor2': {'name': None,   'costs_year': 2020},
        },
        'technical': {
            'mode': 'environmental',
            'method': 'weighted_sum',
            'token': '96e53647b34ef4fb37b6958e9285eff9938ad7ce',
            'budget_tol': 1e-4,
            'formulation': 'milp',
            'name_solver': 'gurobi',
            'beta_obj': BETA_OBJ_DEFAULT,
            'verbose': True,
            'plot_dispatch': True,
            'weight_sweep': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        },
    }
    sols, summaries = run_case(case)

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(summaries)
    summary_path = os.path.join(
        output_dir, "summary_{}.csv".format(case['technical']['method']))
    df.to_csv(summary_path, index=False)    

    if case['technical']['method'] in ('weighted_sum', 'augmecon'):
        path = plot_pareto(
            sols,
            mode=case['technical']['mode'],
            output_dir=output_dir,
            filename='pareto_{}.png'.format(
                case['technical']['method']),
        )

    for s in summaries:
        print("method={}  weight={}  J_eco={:.4g}  J_env={:.4g}".format(
            s['method'],
            s.get('weight', 'n/a'),
            s['objectives']['economic'],
            s['objectives']['environmental']))
        if s['method'] == 'SOO':
            print("-" * 40)
            for k, v in s.items():
                print(f"{k:>24}: {v}")       