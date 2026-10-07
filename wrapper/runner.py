# runner.py

import math
import time
import warnings

import numpy as np
import numpy_financial as npf

from auxiliary_functions import (
    fetch_profile,
    load_price_from_csv,
    load_technology_parameters,
    BETA_OBJ_DEFAULT,
)
from shipp.components import Production, Storage
from shipp.timeseries import TimeSeries
from shipp.kernel_pyomo import solve_lp_pyomo_sizing


def run_case(case):
    """Run a single sizing/dispatch case.

    Args:
        case (dict): flat case description.

    Returns:
        (os_result, summary) where summary is None when case['summarize'] is False.
    """
    n = case['n']
    dt = case.get('dt', 1.0)
    n_year = case['n_year']

    # --- profiles ---
    wind_ts = fetch_profile(
        'wind', case['lat'], case['lon'],
        case['date_from'], case['date_to'], case['token'],
        height=case['wind_height'], turbine=case['wind_turbine'])
    pv_ts = fetch_profile(
        'pv', case['lat'], case['lon'],
        case['date_from'], case['date_to'], case['token'],
        tilt=case['pv_tilt'], azim=case['pv_azim'])
    price_ts = load_price_from_csv(
        case['price_iso3'], case['price_year'], case['price_csv'])

    wind_ts = TimeSeries(wind_ts.data[:n], dt)
    pv_ts = TimeSeries(pv_ts.data[:n], dt)
    price_ts = TimeSeries(price_ts.data[:n], dt)

    tech = load_technology_parameters(
        case['tech_csv'], case['country'], case['param_year'])

    # --- assets ---
    wind_life = tech[('wind', 'lifetime')]
    pv_life = tech[('pv', 'lifetime')]
    lfp_life = tech[('lfp', 'lifetime')]

    wind_nominal = tech[('wind', 'capex')]
    pv_nominal = tech[('pv', 'capex')]
    lfp_e_nominal = tech[('lfp', 'capex')]

    wind = Production(
        power_ts=wind_ts,
        p_cost=wind_nominal * n_year / wind_life,
        p_max=None,
        p_max_is_decision=True,
        opex_fix=tech[('wind', 'opex')],
        opex_var=0.0,
    )
    pv = Production(
        power_ts=pv_ts,
        p_cost=pv_nominal * n_year / pv_life,
        p_max=None,
        p_max_is_decision=True,
        opex_fix=tech[('pv', 'opex')],
        opex_var=0.0,
    )
    eff_half = math.sqrt(tech[('lfp', 'eff_rt_ac')])
    lfp = Storage(
        e_cap=None,
        p_cap=None,
        eff_in=eff_half,
        eff_out=eff_half,
        p_cost=0.0,
        e_cost=lfp_e_nominal * n_year / lfp_life,
        soc_min=0.0,
        soc_max=1.0,
        lifetime=int(lfp_life),
        opex_fix=0.0,
        opex_var=0.0,
        duration=tech[('lfp', 'duration')],
    )

    null_prod = Production(
        power_ts=TimeSeries([0.0] * n, dt), p_cost=1.0, p_max=0.0)
    null_stor = Storage(
        e_cap=0, p_cap=0, eff_in=1.0, eff_out=1.0,
        p_cost=0.0, e_cost=0.0, duration=1.0)

    nominal = {
        'wind_per_mw': wind_nominal,
        'pv_per_mw': pv_nominal,
        'lfp_per_mwh': lfp_e_nominal,
    }

    # --- budget ---
    budget = case.get('budget')
    if budget is not None:
        tol = case.get('budget_tol', 1e-4)
        i_max = budget * (1 + tol)
        i_min = budget * (1 - tol)
        nominal_costs = dict(
            prod1=nominal['wind_per_mw'],
            prod2=nominal['pv_per_mw'],
            stor1_p=0.0,
            stor1_e=nominal['lfp_per_mwh'],
            stor2_p=0.0,
            stor2_e=0.0,
        )
    else:
        i_max = None
        i_min = None
        nominal_costs = None

    # --- solve ---
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        os_result = solve_lp_pyomo_sizing(
            price_ts=price_ts,
            prod1=wind, prod2=pv,
            prof1_unit=wind_ts, prof2_unit=pv_ts,
            stor1=lfp, stor2=null_stor,
            discount_rate=case['discount_rate'], n_year=n_year,
            p_grid_max=case['p_grid_max'], n=n,
            i_max=i_max, i_min=i_min,
            nominal_costs=nominal_costs,
            beta_obj=case.get('beta_obj', BETA_OBJ_DEFAULT),
            options=dict(
                name_solver=case.get('name_solver', 'gurobi'),
                formulation=case.get('formulation', 'milp')),
        )
    t_wall = time.perf_counter() - t0

    if not case.get('summarize', True):
        return os_result, None

    # --- summary ---
    x_wind = os_result.production_list[0].p_max
    x_pv = os_result.production_list[1].p_max
    p_stor = os_result.storage_list[0].p_cap
    e_stor = os_result.storage_list[0].e_cap

    raw = wind_ts.data * x_wind + pv_ts.data * x_pv
    delivered = os_result.power_out.data.sum()
    curtailed = os_result.p_curtail.data.sum()
    losses = sum(l.sum() for l in os_result.losses)
    charge = -np.minimum(os_result.storage_p[0].data, 0).sum()
    discharge = np.maximum(os_result.storage_p[0].data, 0).sum()

    capex_effective = (
        wind.p_cost * x_wind
        + pv.p_cost * x_pv
        + lfp.e_cost * e_stor
    )
    capex_nominal = (
        nominal['wind_per_mw'] * x_wind
        + nominal['pv_per_mw'] * x_pv
        + nominal['lfp_per_mwh'] * e_stor
    )
    opex_total = wind.opex_fix * x_wind + pv.opex_fix * x_pv
    factor = npf.npv(case['discount_rate'], np.ones(n_year)) - 1

    summary = {
        'formulation': case.get('formulation', 'milp'),
        'solver': case.get('name_solver', 'gurobi'),
        'wall_time_s': t_wall,
        'solver_time_s': getattr(os_result, 'time', None),
        'warnings': [str(w.message) for w in caught],

        'x_wind_mw': x_wind,
        'x_pv_mw': x_pv,
        'p_stor_mw': p_stor,
        'e_stor_mwh': e_stor,

        'raw_mwh': raw.sum(),
        'delivered_mwh': delivered,
        'curtailed_mwh': curtailed,
        'curtail_frac': curtailed / raw.sum() if raw.sum() > 0 else 0.0,
        'losses_mwh': losses,
        'charge_mwh': charge,
        'discharge_mwh': discharge,
        'rte_realized': discharge / charge if charge > 0 else None,

        'capex_effective_meur': capex_effective * 1e-6,
        'capex_nominal_meur': capex_nominal * 1e-6,
        'opex_discounted_meur': factor * opex_total * 1e-6,

        'revenue_meur': os_result.revenue * 1e-6,
        'npv_objective_meur': os_result.npv_objective,
        'npv_legacy_meur': os_result.npv,
                'npv_per_capex': (
            os_result.npv_objective
            / (capex_nominal * 1e-6)
            if capex_nominal > 0 else None
        ),

        'loss_check_ok': os_result.check_losses(1e-4),
    }
    return os_result, summary


if __name__ == '__main__':
    case = {
        'token': '96e53647b34ef4fb37b6958e9285eff9938ad7ce',
        'lat': 52.52, 'lon': 4.90,
        'date_from': '2023-01-01', 'date_to': '2024-01-01',
        'price_iso3': 'NLD', 'price_year': 2023,
        'price_csv': 'ember_prices.csv',
        'tech_csv': 'technology_parameters.csv',
        'country': 'DNK', 'param_year': 2025,
        'n': 24 * 365, 'n_year': 25, 'discount_rate': 0.03,
        'p_grid_max': 200,
        'budget': 300e6, 'budget_tol': 1e-4,
        'wind_height': 90, 'wind_turbine': 'Vestas V150 4500',
        'pv_tilt': 35, 'pv_azim': 180,
        'formulation': 'milp', 'name_solver': 'gurobi',
        'beta_obj': BETA_OBJ_DEFAULT,
        'summarize': True,
    }
    os_result, summary = run_case(case)
    if summary is not None:
        for k, v in summary.items():
            print(f"{k:>24}: {v}")