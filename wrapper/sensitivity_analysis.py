# sensitivity_analysis.py

import itertools
import time

import pandas as pd

from runner import run_case
from auxiliary_functions import (
    plot_sweep_decision_vars,
    plot_sweep_npv_per_capex,
    plot_npv_per_capex_vs_grid_cap,
)


BASE = {
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
    'summarize': True,
}


SWEEP_BUDGET = {
    'budget': [50e6, 100e6, 200e6, 300e6],
    'p_grid_max': [50, 80, 100, 120, 150, 200, 250],
}

SWEEP_GRID_ONLY = {
    'p_grid_max': [50, 80, 100, 120, 150, 200, 250],
}


def run_sweep(base=BASE, sweep=SWEEP_BUDGET, tag=None):
    keys = list(sweep.keys())
    grid = list(itertools.product(*(sweep[k] for k in keys)))

    rows = []
    total = len(grid)
    for i, values in enumerate(grid, 1):
        case = dict(base)
        for k, v in zip(keys, values):
            case[k] = v

        label = ', '.join(f"{k}={v}" for k, v in zip(keys, values))
        print(f"[{i}/{total}] {label}", end=' ... ', flush=True)

        t0 = time.perf_counter()
        try:
            _, summary = run_case(case)
        except Exception as exc:
            dt_i = time.perf_counter() - t0
            print(f"error ({dt_i:.1f}s)")
            rows.append({
                'tag': tag,
                **{k: v for k, v in zip(keys, values)},
                'status': 'error',
                'error': f"{type(exc).__name__}: {exc}",
                'wall_s': dt_i,
            })
            continue

        dt_i = time.perf_counter() - t0
        print(f"{dt_i:.1f}s")

        row = {
            'tag': tag,
            **{k: v for k, v in zip(keys, values)},
            'status': 'ok',
        }
        if summary is not None:
            row.update(summary)
        row['wall_s'] = dt_i
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    run_budget = True
    run_grid_only = True
    visualize = True

    frames = []

    if run_budget:
        df_budget = run_sweep(base=BASE, sweep=SWEEP_BUDGET, tag='budget')
        frames.append(df_budget)

    if run_grid_only:
        df_grid = run_sweep(
            base={**BASE, 'budget': None},
            sweep=SWEEP_GRID_ONLY,
            tag='grid_only')
        frames.append(df_grid)

    if not frames:
        raise SystemExit("Nothing to run; enable at least one of "
                         "run_budget, run_grid_only.")

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df.to_csv('sensitivity.csv', index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 50)
    print(df[[
        'tag', 'budget', 'p_grid_max', 'status',
        'x_wind_mw', 'x_pv_mw', 'p_stor_mw', 'e_stor_mwh',
        'capex_nominal_meur', 'npv_objective_meur', 'npv_per_capex',
        'curtail_frac', 'rte_realized',
    ]])

    if visualize and run_budget:
        plot_sweep_decision_vars(
            df_budget, ['budget', 'p_grid_max'],
            title="Decision variables vs budget")
        plot_sweep_npv_per_capex(
            df_budget, ['budget', 'p_grid_max'],
            title="NPV / nominal capex vs budget")

    if visualize and run_grid_only:
        plot_npv_per_capex_vs_grid_cap(
            df_grid,
            title="Budgetless: NPV / CAPEX vs grid cap")