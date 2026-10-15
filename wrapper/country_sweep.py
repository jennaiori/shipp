# country_sweep.py

import itertools
import time

import pandas as pd

from runner import run_case

COUNTRY_SITES = {
    'AUT': (47.5, 14.5),
    'BEL': (50.8, 4.5),
    'CZE': (49.8, 15.5),
    'DNK': (56.0, 9.5),
    'EST': (58.7, 25.5),
    'FIN': (62.5, 26.0),
    'FRA': (47.0, 2.5),
    'DEU': (51.2, 10.4),
    'GRC': (39.0, 22.0),
    'HUN': (47.0, 19.5),
    'ITA': (42.5, 12.5),
    'LVA': (56.9, 24.9),
    'LTU': (55.3, 23.9),
    'LUX': (49.8, 6.1),
    'NLD': (52.3, 5.3),
    'NOR': (60.5, 8.5),
    'POL': (52.0, 19.5),
    'PRT': (39.6, -8.0),
    'ROU': (45.9, 25.0),
    'SVK': (48.7, 19.5),
    'SVN': (46.1, 14.8),
    'ESP': (40.0, -3.5),
    'SWE': (60.0, 15.5),
    'CHE': (46.8, 8.2),
}


BASE_TEMPLATE = {
    'price_csv': 'ember_prices.csv',
    'p_grid_max': 300,
    'budget': None,
    'n': 24 * 365,
    'n_year': 25,
    'discount_rate': 0.08,
    'date_from': '2023-01-01',
    'date_to': '2024-01-01',
}


TECHS = {
    'tech_csv': 'technology_parameters.csv',
    'inflation_csv': 'inflation.csv',
    'pv_costs_year': 2025,
    'pv_tilt': 35, 'pv_azim': 180,
    'wind_costs_year': 2020,
    'wind_height': 90, 'wind_turbine': 'Vestas V150 4500',
    'stor_1_costs_year': 2020,
    'stor_2_costs_year': 2020,
}


TECHNICAL = {
    'token': '96e53647b34ef4fb37b6958e9285eff9938ad7ce',
    'budget_tol': 1e-4,
    'formulation': 'milp',
    'name_solver': 'gurobi',
    'summarize': True,
}


def make_case(country, price_year):
    lat, lon = COUNTRY_SITES[country]
    case = {
        'scenario': {
            **BASE_TEMPLATE,
            'lat': lat, 'lon': lon,
            'price_iso3': country,
            'price_year': price_year,
        },
        'techs': dict(TECHS),
        'technical': dict(TECHNICAL),
    }
    return case


def run_one_country(country, price_year):
    case = make_case(country, price_year)
    t0 = time.perf_counter()
    try:
        _, summary = run_case(case)
    except Exception as exc:
        dt_i = time.perf_counter() - t0
        print(f"error ({dt_i:.1f}s)")
        return {
            'country': country,
            'price_year': price_year,
            'status': 'error',
            'error': f"{type(exc).__name__}: {exc}",
            'wall_s': dt_i,
        }

    dt_i = time.perf_counter() - t0
    print(f"{dt_i:.1f}s")

    row = {
        'country': country,
        'price_year': price_year,
        'status': 'ok',
    }
    if summary is not None:
        row.update(summary)
    row['wall_s'] = dt_i
    return row


def run_sweep(price_years, countries=None):
    if countries is None:
        countries = sorted(COUNTRY_SITES.keys())
    grid = list(itertools.product(countries, price_years))

    rows = []
    total = len(grid)
    for i, (country, year) in enumerate(grid, 1):
        print(f"[{i}/{total}] country={country} year={year}",
              end=' ... ', flush=True)
        rows.append(run_one_country(country, year))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    price_years = [2023]
    df = run_sweep(price_years=price_years)
    df.to_csv('sensitivity.csv', index=False)

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 50)
    print(df[[
        'country', 'price_year', 'status',
        'x_wind_mw', 'x_pv_mw', 'p_stor_mw', 'e_stor_mwh',
        'capex_nominal_meur', 'npv_objective_meur', 'npv_per_capex',
        'curtail_frac', 'rte_realized',
    ]])

    ok = df[df['status'] == 'ok'].copy()
    if len(ok):
        fig, ax = None, None
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ok = ok.sort_values('npv_per_capex', ascending=False)
        ax.bar(ok['country'], ok['npv_per_capex'])
        ax.set_xlabel("Country")
        ax.set_ylabel("NPV / nominal capex [-]")
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        plt.show()