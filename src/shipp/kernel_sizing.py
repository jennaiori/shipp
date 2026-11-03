"""Sizing kernel. Build and solve the integrated sizing/dispatch model."""

import warnings
import numpy as np
import numpy_financial as npf
import pyomo.environ as pyo
import time

from shipp.components import Storage, Production, OpSchedule
from shipp.timeseries import TimeSeries


def build_sizing_model(
    price_eco: TimeSeries,
    price_env: TimeSeries,
    prod1: Production,
    prod2: Production,
    prof1_unit: TimeSeries,
    prof2_unit: TimeSeries,
    stor1: Storage,
    stor2: Storage,
    discount_rate_eco: float,
    discount_rate_env: float,
    n_year: int,
    p_grid_max: float,
    n: int,
    p_min=0.0,
    dp_min: float = None,
    dp_max: float = None,
    i_max: float = None,
    i_min: float = None,
    nominal_costs: dict = None,
    beta_obj: float = 1e-4,
    options: dict = None,
) -> tuple[pyo.ConcreteModel, dict]:
    """Build the sizing/dispatch Pyomo model without an objective.

    Args:
        price_eco, price_env: Day-ahead price and marginal displacement
            factor series.
        prod1, prod2: Production assets. ``p_max`` may be None (decision
            variable) or a float (fixed capacity, 0 excludes the asset).
        prof1_unit, prof2_unit: Per-unit production profiles.
        stor1, stor2: Storage assets. ``p_cap`` may be None (decision
            variable) or a float. ``duration`` must be set.
        discount_rate, n_year: NPV parameters.
        p_grid_max: Maximum grid connection size [MW].
        n: Number of time steps.
        p_min: Minimum grid connection size, scalar or length-n array.
        dp_min, dp_max: Ramp limits [MW/h].
        i_max, i_min: Budget bounds [currency].
        nominal_costs: Optional per-unit costs used only for the budget
            constraints. Keys 'prod1', 'prod2', 'stor1_p', 'stor1_e',
            'stor2_p', 'stor2_e'.
        beta_obj: Curtailment penalty in the objective.
        options: Model options. Keys 'formulation', 'epsilon', 'verbose',
            'return_duals', 'e_start1', 'e_start2', 'name_solver'.

    Returns:
        (model, meta). ``meta`` carries scalar parameters, the resolved
        environmental costs (may be None where missing), the list of
        missing environmental fields, and references to the four assets
        and the two price series.

    Raises:
        AssertionError: on malformed inputs.
        ValueError: on an unrecognised ``p_min`` type.
    """
    dt = price_eco.dt
    assert dt == price_env.dt
    assert dt == prof1_unit.dt
    assert dt == prof2_unit.dt
    assert n <= len(price_eco.data)
    assert n <= len(price_env.data)
    assert n <= len(prof1_unit.data)
    assert n <= len(prof2_unit.data)

    # --- options ---
    formulation = 'milp'
    epsilon = 1e-3
    verbose = False
    return_duals = False
    e_start1 = None
    e_start2 = None
    name_solver = 'gurobi'

    if options is not None:
        if 'formulation' in options:
            formulation = options['formulation']
            assert formulation in ('lp_alt', 'lp', 'milp')
        if 'epsilon' in options:
            epsilon = options['epsilon']
            assert isinstance(epsilon, (float, int))
        if 'verbose' in options:
            verbose = options['verbose']
            assert isinstance(verbose, bool)
        if 'return_duals' in options:
            return_duals = options['return_duals']
            assert isinstance(return_duals, bool)
        if 'e_start1' in options:
            e_start1 = options['e_start1']
            assert isinstance(e_start1, (float, int))
        if 'e_start2' in options:
            e_start2 = options['e_start2']
            assert isinstance(e_start2, (float, int))
        if 'name_solver' in options:
            name_solver = options['name_solver']

    # --- environmental cost resolution ---
    env_missing = []
    env_fields = {
        'prod1': ('p_cost_environmental',),
        'prod2': ('p_cost_environmental',),
        'stor1': ('p_cost_environmental', 'e_cost_environmental'),
        'stor2': ('p_cost_environmental', 'e_cost_environmental'),
    }
    assets = {'prod1': prod1, 'prod2': prod2,
              'stor1': stor1, 'stor2': stor2}
    for asset_name, fields in env_fields.items():
        obj = assets[asset_name]
        for field in fields:
            if getattr(obj, field, None) is None:
                env_missing.append(
                    "{} asset '{}' is missing '{}'".format(
                        type(obj).__name__, asset_name, field))
    env_available = (len(env_missing) == 0)

    # --- input validity ---
    assert np.all(np.isfinite(price_eco.data[:n]))
    assert np.all(np.isfinite(price_env.data[:n]))
    assert np.all(np.isfinite(prof1_unit.data[:n]))
    assert np.all(np.isfinite(prof2_unit.data[:n]))
    assert np.all(np.asarray(prof1_unit.data[:n]) >= 0)
    assert np.all(np.asarray(prof2_unit.data[:n]) >= 0)

    assert np.isfinite(prod1.p_cost_economic)
    assert np.isfinite(prod2.p_cost_economic)
    assert np.isfinite(prod1.opex_fix_economic)
    assert np.isfinite(prod2.opex_fix_economic)
    assert np.isfinite(prod1.opex_var_economic)
    assert np.isfinite(prod2.opex_var_economic)

    assert np.isfinite(stor1.p_cost_economic)
    assert np.isfinite(stor2.p_cost_economic)
    assert np.isfinite(stor1.e_cost_economic)
    assert np.isfinite(stor2.e_cost_economic)
    assert np.isfinite(stor1.eff_in) and np.isfinite(stor1.eff_out)
    assert np.isfinite(stor2.eff_in) and np.isfinite(stor2.eff_out)

    assert np.isfinite(dt)
    assert 0 <= discount_rate_eco <= 1
    assert 0 <= discount_rate_env <= 1

    assert stor1.duration is not None and np.isfinite(stor1.duration) and stor1.duration > 0
    assert stor2.duration is not None and np.isfinite(stor2.duration) and stor2.duration > 0

    if prod1.p_max is not None:
        assert np.isfinite(prod1.p_max)
    if prod2.p_max is not None:
        assert np.isfinite(prod2.p_max)
    if stor1.p_cap is not None:
        assert np.isfinite(stor1.p_cap)
    if stor2.p_cap is not None:
        assert np.isfinite(stor2.p_cap)

    if i_max is not None:
        assert np.isfinite(i_max) and i_max >= 0
    if i_min is not None:
        assert np.isfinite(i_min) and i_min >= 0
    if i_max is not None and i_min is not None:
        assert i_min <= i_max
    if dp_min is not None:
        assert np.isfinite(dp_min) and dp_min <= 0
    if dp_max is not None:
        assert np.isfinite(dp_max) and dp_max >= 0

    assert prod1.p_max is not None or prod1.p_cost_economic > 0
    assert prod2.p_max is not None or prod2.p_cost_economic > 0
    assert stor1.p_cap is not None or stor1.p_cost_economic > 0 or stor1.e_cost_economic > 0
    assert stor2.p_cap is not None or stor2.p_cost_economic > 0 or stor2.e_cost_economic > 0

    if isinstance(p_min, (np.ndarray, list)):
        assert len(p_min) >= n
        p_min_vec = np.asarray(p_min[:n]).reshape(n,)
    elif isinstance(p_min, (float, int)):
        p_min_vec = float(p_min) * np.ones(n)
    else:
        raise ValueError(
            "p_min must be a float, int, list or numpy.ndarray")

    # --- model ---
    model = pyo.ConcreteModel()

    # Design variables
    if prod1.p_max is None:
        model.x1 = pyo.Var(domain=pyo.NonNegativeReals)
    else:
        model.x1 = pyo.Var(bounds=(prod1.p_max, prod1.p_max))
    if prod2.p_max is None:
        model.x2 = pyo.Var(domain=pyo.NonNegativeReals)
    else:
        model.x2 = pyo.Var(bounds=(prod2.p_max, prod2.p_max))
    if stor1.p_cap is None:
        model.p_cap1 = pyo.Var(domain=pyo.NonNegativeReals)
    else:
        model.p_cap1 = pyo.Var(bounds=(stor1.p_cap, stor1.p_cap))
    if stor2.p_cap is None:
        model.p_cap2 = pyo.Var(domain=pyo.NonNegativeReals)
    else:
        model.p_cap2 = pyo.Var(bounds=(stor2.p_cap, stor2.p_cap))

    model.e_cap1 = pyo.Expression(expr=stor1.duration * model.p_cap1)
    model.e_cap2 = pyo.Expression(expr=stor2.duration * model.p_cap2)

    # Operation variables
    model.vec_n = pyo.Set(initialize=list(range(n)))
    model.vec_np1 = pyo.Set(initialize=list(range(n + 1)))
    model.vec_nm1 = pyo.Set(initialize=list(range(n - 1)))

    if formulation == 'lp_alt':
        model.p_vec1 = pyo.Var(model.vec_n)
        model.p_vec2 = pyo.Var(model.vec_n)
    else:
        model.p_vec1_charge = pyo.Var(model.vec_n, domain=pyo.NonNegativeReals)
        model.p_vec1_discharge = pyo.Var(model.vec_n, domain=pyo.NonNegativeReals)
        model.p_vec2_charge = pyo.Var(model.vec_n, domain=pyo.NonNegativeReals)
        model.p_vec2_discharge = pyo.Var(model.vec_n, domain=pyo.NonNegativeReals)

    model.e_vec1 = pyo.Var(model.vec_np1, domain=pyo.NonNegativeReals)
    model.e_vec2 = pyo.Var(model.vec_np1, domain=pyo.NonNegativeReals)
    model.p_cur = pyo.Var(model.vec_n, domain=pyo.NonNegativeReals)

    if formulation == 'milp':
        model.bin1 = pyo.Var(model.vec_n, within=pyo.Binary)
        model.bin2 = pyo.Var(model.vec_n, within=pyo.Binary)

    if formulation == 'lp_alt':
        def p_stor(t):
            return model.p_vec1[t] + model.p_vec2[t]
    else:
        def p_stor(t):
            return (model.p_vec1_discharge[t] - model.p_vec1_charge[t]
                    + model.p_vec2_discharge[t] - model.p_vec2_charge[t])

    # --- revenue and cost expressions ---
    factor_eco = npf.npv(discount_rate_eco, np.ones(n_year)) - 1
    factor_env = npf.npv(discount_rate_env, np.ones(n_year)) - 1

    revenue_eco = pyo.Expression(expr=365 * 24 / n * factor_eco * sum(
        (price_eco.data[t] + beta_obj)
        * (model.x1 * prof1_unit.data[t]
        + model.x2 * prof2_unit.data[t]
        + p_stor(t)
        - model.p_cur[t])
        for t in model.vec_n))
    revenue_env = pyo.Expression(expr=365 * 24 / n * factor_env * sum(
        (price_env.data[t] + beta_obj)
        * (model.x1 * prof1_unit.data[t]
        + model.x2 * prof2_unit.data[t]
        + p_stor(t)
        - model.p_cur[t])
        for t in model.vec_n))
    model.revenue_eco = revenue_eco
    model.revenue_env = revenue_env

    opex_eco = pyo.Expression(expr=- factor_eco * (
        prod1.opex_fix_economic * model.x1
        + prod2.opex_fix_economic * model.x2))
    model.opex_eco = opex_eco

    capex_eco = pyo.Expression(expr=(
        - prod1.p_cost_economic * model.x1
        - prod2.p_cost_economic * model.x2
        - stor1.p_cost_economic * model.p_cap1
        - stor1.e_cost_economic * model.e_cap1
        - stor2.p_cost_economic * model.p_cap2
        - stor2.e_cost_economic * model.e_cap2))
    model.capex_eco = capex_eco

    model.J_eco = pyo.Expression(expr=revenue_eco + opex_eco + capex_eco)

    if env_available:
        capex_env = pyo.Expression(expr=(
            - prod1.p_cost_environmental * model.x1
            - prod2.p_cost_environmental * model.x2
            - stor1.p_cost_environmental * model.p_cap1
            - stor1.e_cost_environmental * model.e_cap1
            - stor2.p_cost_environmental * model.p_cap2
            - stor2.e_cost_environmental * model.e_cap2))
        model.capex_env = capex_env
        model.J_env = pyo.Expression(expr=revenue_env + capex_env)

    # --- rules ---
    if formulation == 'lp_alt':
        def rule_p_max1(m, i):
            return m.p_vec1[i] <= m.p_cap1
        def rule_p_min1(m, i):
            return m.p_vec1[i] >= -m.p_cap1
        def rule_p_max2(m, i):
            return m.p_vec2[i] <= m.p_cap2
        def rule_p_min2(m, i):
            return m.p_vec2[i] >= -m.p_cap2
        def rule_e_model_charge1(m, i):
            return m.e_vec1[i+1] - m.e_vec1[i] <= -dt * stor1.eff_in * m.p_vec1[i]
        def rule_e_model_discharge1(m, i):
            return m.e_vec1[i+1] - m.e_vec1[i] <= -dt / stor1.eff_out * m.p_vec1[i]
        def rule_e_model_charge2(m, i):
            return m.e_vec2[i+1] - m.e_vec2[i] <= -dt * stor2.eff_in * m.p_vec2[i]
        def rule_e_model_discharge2(m, i):
            return m.e_vec2[i+1] - m.e_vec2[i] <= -dt / stor2.eff_out * m.p_vec2[i]
    else:
        def rule_p_max1(m, i):
            return m.p_vec1_charge[i] <= m.p_cap1
        def rule_p_min1(m, i):
            return m.p_vec1_discharge[i] <= m.p_cap1
        def rule_p_max2(m, i):
            return m.p_vec2_charge[i] <= m.p_cap2
        def rule_p_min2(m, i):
            return m.p_vec2_discharge[i] <= m.p_cap2
        def rule_e_model1(m, i):
            return (m.e_vec1[i+1] - m.e_vec1[i]
                    == dt * stor1.eff_in * m.p_vec1_charge[i]
                    - dt / stor1.eff_out * m.p_vec1_discharge[i])
        def rule_e_model2(m, i):
            return (m.e_vec2[i+1] - m.e_vec2[i]
                    == dt * stor2.eff_in * m.p_vec2_charge[i]
                    - dt / stor2.eff_out * m.p_vec2_discharge[i])

    def rule_e_max1(m, i):
        return m.e_vec1[i] <= m.e_cap1 * stor1.soc_max
    def rule_e_min1(m, i):
        return m.e_vec1[i] >= m.e_cap1 * stor1.soc_min
    def rule_e_max2(m, i):
        return m.e_vec2[i] <= m.e_cap2 * stor2.soc_max
    def rule_e_min2(m, i):
        return m.e_vec2[i] >= m.e_cap2 * stor2.soc_min

    def rule_p_tot_min(m, i):
        return (m.x1 * prof1_unit.data[i]
                + m.x2 * prof2_unit.data[i]
                + p_stor(i) - m.p_cur[i]) >= p_min_vec[i]

    def rule_p_tot_max_curt(m, i):
        return (m.x1 * prof1_unit.data[i]
                + m.x2 * prof2_unit.data[i]
                + p_stor(i) - m.p_cur[i]) <= p_grid_max

    def rule_p_cur_lim(m, i):
        return m.p_cur[i] <= (m.x1 * prof1_unit.data[i]
                              + m.x2 * prof2_unit.data[i])

    model.e_start_end1 = pyo.Constraint(expr=model.e_vec1[0] == model.e_vec1[n])
    model.e_start_end2 = pyo.Constraint(expr=model.e_vec2[0] == model.e_vec2[n])
    if e_start1 is not None:
        model.e_fix_start1 = pyo.Constraint(expr=model.e_vec1[0] == e_start1)
    if e_start2 is not None:
        model.e_fix_start2 = pyo.Constraint(expr=model.e_vec2[0] == e_start2)

    model.p_min1 = pyo.Constraint(model.vec_n, rule=rule_p_min1)
    model.p_max1 = pyo.Constraint(model.vec_n, rule=rule_p_max1)
    model.e_max1 = pyo.Constraint(model.vec_n, rule=rule_e_max1)
    model.e_min1 = pyo.Constraint(model.vec_n, rule=rule_e_min1)
    model.p_min2 = pyo.Constraint(model.vec_n, rule=rule_p_min2)
    model.p_max2 = pyo.Constraint(model.vec_n, rule=rule_p_max2)
    model.e_max2 = pyo.Constraint(model.vec_n, rule=rule_e_max2)
    model.e_min2 = pyo.Constraint(model.vec_n, rule=rule_e_min2)

    if formulation == 'lp_alt':
        model.e_model_charge1 = pyo.Constraint(model.vec_n, rule=rule_e_model_charge1)
        model.e_model_discharge1 = pyo.Constraint(model.vec_n, rule=rule_e_model_discharge1)
        model.e_model_charge2 = pyo.Constraint(model.vec_n, rule=rule_e_model_charge2)
        model.e_model_discharge2 = pyo.Constraint(model.vec_n, rule=rule_e_model_discharge2)
    else:
        model.e_model1 = pyo.Constraint(model.vec_n, rule=rule_e_model1)
        model.e_model2 = pyo.Constraint(model.vec_n, rule=rule_e_model2)
        if formulation == 'milp':
            M = (float(p_grid_max) if np.isfinite(p_grid_max) else
                 max(float(stor1.p_cap or 0), float(stor2.p_cap or 0), 1.0))
            model.bin_charge1 = pyo.Constraint(
                model.vec_n,
                rule=lambda m, i: m.p_vec1_charge[i] <= M * (1 - m.bin1[i]))
            model.bin_discharge1 = pyo.Constraint(
                model.vec_n,
                rule=lambda m, i: m.p_vec1_discharge[i] <= M * m.bin1[i])
            model.bin_charge2 = pyo.Constraint(
                model.vec_n,
                rule=lambda m, i: m.p_vec2_charge[i] <= M * (1 - m.bin2[i]))
            model.bin_discharge2 = pyo.Constraint(
                model.vec_n,
                rule=lambda m, i: m.p_vec2_discharge[i] <= M * m.bin2[i])

    if dp_min is not None:
        def rule_dp_tot_min(m, i):
            return (p_stor(i+1) - p_stor(i)
                    - (m.p_cur[i+1] - m.p_cur[i])
                    >= dp_min
                    - (m.x1 * prof1_unit.data[i+1] - m.x1 * prof1_unit.data[i])
                    - (m.x2 * prof2_unit.data[i+1] - m.x2 * prof2_unit.data[i]))
        model.dp_tot_min = pyo.Constraint(model.vec_nm1, rule=rule_dp_tot_min)

    if dp_max is not None:
        def rule_dp_tot_max(m, i):
            return (p_stor(i+1) - p_stor(i)
                    - (m.p_cur[i+1] - m.p_cur[i])
                    <= dp_max
                    - (m.x1 * prof1_unit.data[i+1] - m.x1 * prof1_unit.data[i])
                    - (m.x2 * prof2_unit.data[i+1] - m.x2 * prof2_unit.data[i]))
        model.dp_tot_max = pyo.Constraint(model.vec_nm1, rule=rule_dp_tot_max)

    model.p_tot_min = pyo.Constraint(model.vec_n, rule=rule_p_tot_min)
    model.p_tot_max_curt = pyo.Constraint(model.vec_n, rule=rule_p_tot_max_curt)
    model.p_cur_lim = pyo.Constraint(model.vec_n, rule=rule_p_cur_lim)

    # Budget
    if nominal_costs is not None:
        budget_expr = (
            - nominal_costs['prod1'] * model.x1
            - nominal_costs['prod2'] * model.x2
            - nominal_costs['stor1_p'] * model.p_cap1
            - nominal_costs['stor1_e'] * model.e_cap1
            - nominal_costs['stor2_p'] * model.p_cap2
            - nominal_costs['stor2_e'] * model.e_cap2)
    else:
        budget_expr = capex_eco

    if i_max is not None:
        model.i_cap_max = pyo.Constraint(expr=-budget_expr <= i_max)
    if i_min is not None:
        model.i_cap_min = pyo.Constraint(expr=-budget_expr >= i_min)

    if return_duals:
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    meta = {
        'dt': dt, 'n': n, 'n_year': n_year,
        'discount_rate_eco': discount_rate_eco,
        'discount_rate_env': discount_rate_env,
        'name_solver': name_solver,
        'formulation': formulation,
        'verbose': verbose,
        'return_duals': return_duals,
        'p_grid_max': p_grid_max,
        'prod1': prod1, 'prod2': prod2,
        'stor1': stor1, 'stor2': stor2,
        'prof1_unit': prof1_unit, 'prof2_unit': prof2_unit,
        'price_eco': price_eco, 'price_env': price_env,
        'env_missing': env_missing,
        'env_available': env_available,
    }
    return model, meta

def solve_sizing_SOO(model, meta, dimension='economic'):
    if dimension not in ('economic', 'environmental'):
        raise ValueError(
            "dimension must be 'economic' or 'environmental', got '{}'".format(
                dimension))
    if dimension == 'environmental' and not meta['env_available']:
        raise ValueError(
            "Environmental SOO requested but the following assets "
            "have no environmental cost basis:\n  - "
            + "\n  - ".join(meta['env_missing']))

    if hasattr(model, 'obj'):
        model.del_component(model.obj)
    expr = model.J_eco if dimension == 'economic' else model.J_env
    model.obj = pyo.Objective(expr=expr, sense=pyo.maximize)

    opt = pyo.SolverFactory(meta['name_solver'])
    if 'gurobi' in meta['name_solver']:
        opt.options['MIPGap'] = 1e-6
        opt.options['Seed'] = 0
        opt.options['Threads'] = 1
        opt.options['StartNodeLimit'] = -3
        opt.options['NumericFocus'] = 3
    results = opt.solve(model, tee=meta['verbose'])
    if (results.solver.status is not pyo.SolverStatus.ok
            or results.solver.termination_condition
            is not pyo.TerminationCondition.optimal):
        raise RuntimeError(
            "Solver did not return an optimal solution: status={}, "
            "termination={}".format(results.solver.status,
                                    results.solver.termination_condition))

    objectives = {'economic': pyo.value(model.J_eco)}
    objectives['environmental'] = (
        pyo.value(model.J_env) if meta['env_available'] else None)

    return {
        'method': 'SOO',
        'dimension': dimension,
        'objectives': objectives,
        'os_result': extract_solution(model, meta),
    }


def solve_sizing_MOO(model, meta, method='weighted_sum', weights=None,
                     n_points=11, verbose=False):
    if not meta['env_available']:
        raise ValueError(
            "MOO requires environmental costs; the following assets "
            "have none:\n  - " + "\n  - ".join(meta['env_missing']))

    if method == 'weighted_sum':
        if weights is None:
            weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        weights = [float(w) for w in weights]
        for w in weights:
            if not (0.0 <= w <= 1.0):
                raise ValueError(
                    "weight sweep values must be in [0, 1], got {}".format(w))

        sol_eco = solve_sizing_SOO(model, meta, dimension='economic')
        scale_eco = abs(sol_eco['objectives']['economic'])
        sol_env = solve_sizing_SOO(model, meta, dimension='environmental')
        scale_env = abs(sol_env['objectives']['environmental'])

        if scale_eco < 1e-12:
            raise ValueError(
                "Economic SOO optimum is zero; weighted sum is undefined.")
        if scale_env < 1e-12:
            raise ValueError(
                "Environmental SOO optimum is zero; weighted sum is undefined.")

        solutions = []
        for w in weights:
            w_eco = float(w)
            w_env = 1.0 - w_eco
            if hasattr(model, 'obj'):
                model.del_component(model.obj)
            model.obj = pyo.Objective(
                expr=w_eco * model.J_eco / scale_eco
                    + w_env * model.J_env / scale_env,
                sense=pyo.maximize)

            if verbose:
                print("  w={:.2f}...".format(w_eco), end=" ", flush=True)
            t_w = time.perf_counter()

            opt = pyo.SolverFactory(meta['name_solver'])
            if 'gurobi' in meta['name_solver']:
                opt.options['MIPGap'] = 1e-6
                opt.options['Seed'] = 0
                opt.options['Threads'] = 1
                opt.options['StartNodeLimit'] = -3
                opt.options['NumericFocus'] = 3
            results = opt.solve(model, tee=meta['verbose'])

            t_solve = time.perf_counter() - t_w

            if (results.solver.status is not pyo.SolverStatus.ok
                    or results.solver.termination_condition
                    is not pyo.TerminationCondition.optimal):
                raise RuntimeError(
                    "Weighted-sum solve did not return an optimal solution: "
                    "status={}, termination={}, weight={}".format(
                        results.solver.status,
                        results.solver.termination_condition,
                        w_eco))

            eco_val = pyo.value(model.J_eco)
            env_val = pyo.value(model.J_env)
            if verbose:
                print("{:.2f} s  J_eco={:.4g}  J_env={:.4g}".format(
                    t_solve, eco_val, env_val))

            solutions.append({
                'method': 'weighted_sum',
                'weight': w_eco,
                'scale_eco': scale_eco,
                'scale_env': scale_env,
                'objectives': {'economic': eco_val, 'environmental': env_val},
                'os_result': extract_solution(model, meta),
            })

    if method == 'augmecon':
        raise NotImplementedError(
            "AUGMECON is not yet implemented.")

    if method != 'weighted_sum' and method != 'augmecon':
        raise ValueError(
            "method must be 'weighted_sum' or 'augmecon', got '{}'".format(method))

    return solutions

def extract_solution(
    model: pyo.ConcreteModel,
    meta: dict,
) -> OpSchedule:
    """Build the OpSchedule from a solved model.

    Raises:
        RuntimeError: if a storage capacity is negative beyond solver
            tolerance.
    """
    n = meta['n']
    dt = meta['dt']
    formulation = meta['formulation']
    prod1 = meta['prod1']
    prod2 = meta['prod2']
    stor1 = meta['stor1']
    stor2 = meta['stor2']
    prof1_unit = meta['prof1_unit']
    prof2_unit = meta['prof2_unit']
    price_eco = meta['price_eco']

    # --- read decision variables ---
    p_cur = [pyo.value(model.p_cur[e]) for e in model.vec_n]

    e_vec1 = [pyo.value(model.e_vec1[e]) for e in model.vec_np1]
    e_vec2 = [pyo.value(model.e_vec2[e]) for e in model.vec_np1]

    p_cap1_raw = pyo.value(model.p_cap1)
    p_cap2_raw = pyo.value(model.p_cap2)
    e_cap1_raw = pyo.value(model.e_cap1)
    e_cap2_raw = pyo.value(model.e_cap2)
    if p_cap1_raw < -1e-6 or e_cap1_raw < -1e-6:
        raise RuntimeError(
            "Negative storage capacity at extract: "
            "p_cap1 = {:.6g}, e_cap1 = {:.6g}".format(p_cap1_raw, e_cap1_raw))
    if p_cap2_raw < -1e-6 or e_cap2_raw < -1e-6:
        raise RuntimeError(
            "Negative storage capacity at extract: "
            "p_cap2 = {:.6g}, e_cap2 = {:.6g}".format(p_cap2_raw, e_cap2_raw))
    p_cap1 = max(p_cap1_raw, 0.0)
    p_cap2 = max(p_cap2_raw, 0.0)
    e_cap1 = max(e_cap1_raw, 0.0)
    e_cap2 = max(e_cap2_raw, 0.0)

    if formulation == 'lp_alt':
        p_vec1 = [pyo.value(model.p_vec1[e]) for e in model.vec_n]
        p_vec2 = [pyo.value(model.p_vec2[e]) for e in model.vec_n]
    else:
        p_vec1 = [pyo.value(model.p_vec1_discharge[e])
                  - pyo.value(model.p_vec1_charge[e]) for e in model.vec_n]
        p_vec2 = [pyo.value(model.p_vec2_discharge[e])
                  - pyo.value(model.p_vec2_charge[e]) for e in model.vec_n]

    x1 = pyo.value(model.x1)
    x2 = pyo.value(model.x2)

    # --- losses ---
    power_losses_bat = []
    power_losses_h2 = []
    for i in range(n):
        power_losses_bat.append(-(e_vec1[i+1] - e_vec1[i] + dt*p_vec1[i])/dt)
        power_losses_h2.append(-(e_vec2[i+1] - e_vec2[i] + dt*p_vec2[i])/dt)

    # --- rebuild Storage / Production with both dimensions ---
    stor1_res = Storage(
        e_cap=e_cap1, p_cap=p_cap1,
        eff_in=stor1.eff_in, eff_out=stor1.eff_out,
        soc_min=stor1.soc_min, soc_max=stor1.soc_max,
        lifetime=stor1.lifetime, duration=stor1.duration,
        p_cost_economic=stor1.p_cost_economic,
        p_cost_environmental=stor1.p_cost_environmental,
        e_cost_economic=stor1.e_cost_economic,
        e_cost_environmental=stor1.e_cost_environmental,
        opex_fix_economic=stor1.opex_fix_economic,
        opex_fix_environmental=stor1.opex_fix_environmental,
        opex_var_economic=stor1.opex_var_economic,
        opex_var_environmental=stor1.opex_var_environmental)
    stor2_res = Storage(
        e_cap=e_cap2, p_cap=p_cap2,
        eff_in=stor2.eff_in, eff_out=stor2.eff_out,
        soc_min=stor2.soc_min, soc_max=stor2.soc_max,
        lifetime=stor2.lifetime, duration=stor2.duration,
        p_cost_economic=stor2.p_cost_economic,
        p_cost_environmental=stor2.p_cost_environmental,
        e_cost_economic=stor2.e_cost_economic,
        e_cost_environmental=stor2.e_cost_environmental,
        opex_fix_economic=stor2.opex_fix_economic,
        opex_fix_environmental=stor2.opex_fix_environmental,
        opex_var_economic=stor2.opex_var_economic,
        opex_var_environmental=stor2.opex_var_environmental)

    # --- headroom check ---
    if meta['p_grid_max'] is not None:
        A_opt = (np.array(prof1_unit.data[:n]) * x1
                 + np.array(prof2_unit.data[:n]) * x2)
        headroom = np.maximum(meta['p_grid_max'] - A_opt, 0.0)
        p_stor_net = np.array(p_vec1) + np.array(p_vec2)
        violation = p_stor_net - headroom - np.array(p_cur)
        max_violation = float(np.max(violation))
        if max_violation > 1e-6:
            warnings.warn(
                "Storage headroom violated by {:.6g} MW in "
                "extract_solution. Optimal dispatch may be infeasible.".format(
                    max_violation),
                RuntimeWarning)

    # --- realized production profiles, curtailment split by availability ---
    prod1_avail = np.array(prof1_unit.data[:n]) * x1
    prod2_avail = np.array(prof2_unit.data[:n]) * x2
    total_avail = prod1_avail + prod2_avail
    frac1 = np.zeros_like(total_avail)
    mask = total_avail > 0
    frac1[mask] = prod1_avail[mask] / total_avail[mask]
    p_cur1 = np.array(p_cur) * frac1
    p_cur2 = np.array(p_cur) - p_cur1

    prod1_res = Production(
        power_ts=TimeSeries(prod1_avail - p_cur1, dt),
        p_max=x1,
        p_cost_economic=prod1.p_cost_economic,
        p_cost_environmental=prod1.p_cost_environmental,
        opex_fix_economic=prod1.opex_fix_economic,
        opex_fix_environmental=prod1.opex_fix_environmental,
        opex_var_economic=prod1.opex_var_economic,
        opex_var_environmental=prod1.opex_var_environmental)
    prod2_res = Production(
        power_ts=TimeSeries(prod2_avail - p_cur2, dt),
        p_max=x2,
        p_cost_economic=prod2.p_cost_economic,
        p_cost_environmental=prod2.p_cost_environmental,
        opex_fix_economic=prod2.opex_fix_economic,
        opex_fix_environmental=prod2.opex_fix_environmental,
        opex_var_economic=prod2.opex_var_economic,
        opex_var_environmental=prod2.opex_var_environmental)

    os_res = OpSchedule(
        production_list=[prod1_res, prod2_res],
        storage_list=[stor1_res, stor2_res],
        production_p=[TimeSeries(prod1_avail, dt),
                      TimeSeries(prod2_avail, dt)],
        storage_p=[TimeSeries(p_vec1, dt),
                   TimeSeries(p_vec2, dt)],
        storage_e=[TimeSeries(e_vec1[:n], dt),
                   TimeSeries(e_vec2[:n], dt)],
        price=price_eco.data[:n],
        p_curtail=TimeSeries(p_cur, dt))

    os_res.get_npv_irr(meta['discount_rate_eco'], meta['n_year'])
    os_res.npv_objective = pyo.value(model.obj) * 1e-6

    os_res.losses = [np.array(power_losses_bat), np.array(power_losses_h2)]

    if not os_res.check_losses(1e-4):
        os_res.check_losses(1e-4, True)
        warnings.warn(
            'Failed error check in extract_solution', RuntimeWarning)

    # --- solve time ---
    if hasattr(model, '_solve_time'):
        os_res.time = model._solve_time
    else:
        os_res.time = None

    # --- duals ---
    os_res.dual_prices = None
    if meta['return_duals']:
        if formulation != 'lp_alt':
            warnings.warn(
                "return_duals is only implemented for formulation='lp_alt'; "
                "skipping dual extraction.",
                RuntimeWarning)
        else:
            try:
                dual_e_min1 = np.array([
                    model.dual.get(model.e_min1[i], 0.0) for i in range(n)])
                dual_e_max1 = np.array([
                    model.dual.get(model.e_max1[i], 0.0) for i in range(n)])
                dual_charge1 = np.array([
                    model.dual.get(model.e_model_charge1[i], 0.0)
                    for i in range(n)])
                dual_discharge1 = np.array([
                    model.dual.get(model.e_model_discharge1[i], 0.0)
                    for i in range(n)])
                os_res.dual_prices = {
                    "dual_e_min1": dual_e_min1,
                    "dual_e_max1": dual_e_max1,
                    "dual_charge1": dual_charge1,
                    "dual_discharge1": dual_discharge1,
                    "dual_energy1": dual_charge1 + dual_discharge1,
                }
            except Exception as exc:
                warnings.warn(
                    "Dual extraction failed: {}".format(exc), RuntimeWarning)
                os_res.dual_prices = None

    return os_res