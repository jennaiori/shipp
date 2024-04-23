"""This module defines kernel functions for shipp.

The functions defined in this module are used to analyze or compute data
for the classes defined in shipp.

Functions:
    solve_lp_pyomo: Build and solve a LP for NPV maximization.

"""

import numpy as np
import numpy_financial as npf

import pyomo.environ as pyo

from shipp.components import Storage, OpSchedule, Production
from shipp.timeseries import TimeSeries

def solve_lp_pyomo(price_ts: TimeSeries, prod_wind: Production,
                    prod_pv: Production, stor1: Storage, stor2: Storage,
                    discount_rate: float, n_year: int,
                    p_min: float | np.ndarray, p_max: float,
                    n: int, name_solver: str = 'mosek') -> OpSchedule:
    """Build and solve a LP for NPV maximization with pyomo.

    This function builds and solves the hybrid sizing and operation
    problem as a linear program, in a short formulation (sf). The
    objective is to minimize the Net Present Value of the plant. In this
    function, the input for the power production represented by two
    Production objects, one for wind and one for solar.

    Params:
        price_ts (TimeSeries): Time series of the price of electricity
            on theday-ahead market [currency/MWh].
        prod_wind (Production): Object representing the power production
            from wind energy system.
        prod_pv (Production): Object representing the power production
            from solar PV system.
        stor1 (Storage): Object describing the battery storage.
        stor2 (Storage): Object describing the hydrogen storage system.
        discount_rate (float): Discount rate for the NPV calculation [-].
        n_year (int): Number of years for the NPV calculation [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational
            schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time
            series do not match, if the length of the power in the
            Production objects is below n.
        RuntimeError: if the optimization algorithm fails to solve the
            problem.
    """

    dt = prod_wind.power.dt

    assert dt == price_ts.dt
    assert dt == prod_pv.power.dt
    assert n <=  len(prod_wind.power.data)
    assert n <=  len(prod_pv.power.data)
    assert n <=  len(price_ts.data)



    power_res = prod_wind.power.data[:n] + prod_pv.power.data[:n]

    p_cost1 = stor1.p_cost
    e_cost1 = stor1.e_cost
    eta1 = stor1.eff_out * stor1.eff_in

    p_cost2 = stor2.p_cost
    e_cost2 = stor2.e_cost
    eta2 = stor2.eff_out * stor2.eff_in


    assert np.all(np.isfinite(power_res))
    assert np.all(np.isfinite(p_min))
    assert np.isfinite(p_max)
    assert np.isfinite(dt)
    assert np.isfinite(eta1)
    assert np.isfinite(eta2)

    if isinstance(p_min, (np.ndarray, list)):
        assert len(p_min) >= n
        p_min_vec = p_min[:n].reshape(n,)
    elif isinstance(p_min, (float, int)):
        p_min_vec = p_min * np.ones((n,))
    else:
        raise ValueError("Input p_min in solve_lp_pyomo must be a float, int,\
                          list or numpy.array")


    #Concrete Model
    model = pyo.ConcreteModel()
    #Decision Variables
    model.vec_n = pyo.Set(initialize=list(range(n)))
    model.vec_np1 = pyo.Set(initialize=list(range(n+1)))

    model.p_vec1 = pyo.Var(model.vec_n)
    model.e_vec1 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals)
    model.p_vec2 = pyo.Var(model.vec_n)
    model.e_vec2 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals)

    if stor1.p_cap == -1 or stor1.p_cap is None:
        model.p_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    else:
        model.p_cap1 = pyo.Var(bounds = (0, stor1.p_cap))

    if stor2.p_cap == -1 or stor2.p_cap is None:
        model.p_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    else:
        model.p_cap2 = pyo.Var(bounds = (0, stor2.p_cap))

    if stor1.e_cap == -1 or stor1.e_cap is None:
        model.e_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    else:
        model.e_cap1 = pyo.Var(bounds = (0, stor1.e_cap))

    if stor2.e_cap == -1 or stor2.e_cap is None:
        model.e_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    else:
        model.e_cap2 = pyo.Var(bounds = (0, stor2.e_cap))

    #Objective
    factor = npf.npv(discount_rate, np.ones(n_year))-1
    model.obj = pyo.Objective(expr=  365 * 24/n*factor* sum([p*(model.p_vec1[n]
                 + model.p_vec2[k]) for p, n, k in zip(price_ts.data[:n],
                model.p_vec1, model.p_vec2)]) - p_cost1*model.p_cap1
                - e_cost1*model.e_cap1 - p_cost2*model.p_cap2
                - e_cost2*model.e_cap2, sense = pyo.maximize)

    # Rule functions for the constraints
    def rule_e_model_charge1(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] <= - dt * model.p_vec1[i]

    def rule_e_model_discharge1(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] <= - dt/eta1 * model.p_vec1[i]

    def rule_p_max1(model, i):
        return model.p_vec1[i] <= model.p_cap1

    def rule_p_min1(model, i):
        return model.p_vec1[i] >= -model.p_cap1

    def rule_e_max1(model, i):
        return model.e_vec1[i] <= model.e_cap1

    def rule_e_model_charge2(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] <= - dt * model.p_vec2[i]

    def rule_e_model_discharge2(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] <= - dt/eta2 * model.p_vec2[i]

    def rule_p_max2(model, i):
        return model.p_vec2[i] <= model.p_cap2

    def rule_p_min2(model, i):
        return model.p_vec2[i] >= -model.p_cap2

    def rule_e_max2(model, i):
        return model.e_vec2[i] <= model.e_cap2

    # def rule_p_tot_min(model, i):
    #   return model.p_vec1[i] + model.p_vec2[i] >= p_min_vec[i] - power_res[i]

    def rule_p_tot_min(model, i):
        return model.p_vec1[i] + model.p_vec2[i] >= p_min_vec[i]- power_res[i]





    def rule_p_tot_max(model, i):
        return model.p_vec1[i] + model.p_vec2[i] <= max(p_max - power_res[i], 0)

    # Constraint for each storage type
    model.e_start_end1 =pyo.Constraint(expr = model.e_vec1[0]==model.e_vec1[n])
    model.e_start_end2 =pyo.Constraint(expr = model.e_vec2[0]==model.e_vec2[n])

    model.e_model_charge1 = pyo.Constraint(model.vec_n,
                                           rule=rule_e_model_charge1)
    model.e_model_discharge1 = pyo.Constraint(model.vec_n,
                                              rule=rule_e_model_discharge1)

    model.p_min1 = pyo.Constraint(model.vec_n, rule=rule_p_min1)
    model.p_max1 = pyo.Constraint(model.vec_n, rule=rule_p_max1)
    model.e_max1 = pyo.Constraint(model.vec_n, rule=rule_e_max1)

    model.e_model_charge2 = pyo.Constraint(model.vec_n,
                                           rule=rule_e_model_charge2)
    model.e_model_discharge2 = pyo.Constraint(model.vec_n,
                                              rule=rule_e_model_discharge2)

    model.p_min2 = pyo.Constraint(model.vec_n, rule=rule_p_min2)
    model.p_max2 = pyo.Constraint(model.vec_n, rule=rule_p_max2)
    model.e_max2 = pyo.Constraint(model.vec_n, rule=rule_e_max2)

    # Global constraints
    model.p_tot_min = pyo.Constraint(model.vec_n, rule=rule_p_tot_min)
    model.p_tot_max = pyo.Constraint(model.vec_n, rule=rule_p_tot_max)

    results = pyo.SolverFactory(name_solver).solve(model)
    # model.display()

    #Check if the problem was solved correclty
    if (results.solver.status is not pyo.SolverStatus.ok) or \
        (results.solver.termination_condition is not 
         pyo.TerminationCondition.optimal):
        raise RuntimeError
    # Do something when the solution in optimal and feasible
    

    e_vec1 = [pyo.value(model.e_vec1[e]) for e in model.e_vec1]
    p_vec1 = [pyo.value(model.p_vec1[e]) for e in model.p_vec1]
    e_cap1 = pyo.value(model.e_cap1)
    p_cap1 = pyo.value(model.p_cap1)

    e_vec2 = [pyo.value(model.e_vec2[e]) for e in model.e_vec2]
    p_vec2 = [pyo.value(model.p_vec2[e]) for e in model.p_vec2]
    e_cap2 = pyo.value(model.e_cap2)
    p_cap2 = pyo.value(model.p_cap2)


    power_res_new = []
    power_losses_bat = []
    power_losses_h2 = []
    for i in range(n):
        power_res_new.append(min(p_max - p_vec1[i] - p_vec2[i],
                             power_res[i]))

        power_losses_bat.append(-(e_vec1[i+1] - e_vec1[i] + dt*p_vec1[i])/dt)
        power_losses_h2.append(-(e_vec2[i+1] - e_vec2[i] + dt*p_vec2[i])/dt)

    stor1_res = Storage(e_cap = e_cap1,
                            p_cap = p_cap1,
                            eff_in = 1,
                            eff_out = eta1,
                            p_cost = stor1.p_cost,
                            e_cost = stor1.e_cost)
    stor2_res = Storage(e_cap = e_cap2,
                            p_cap = p_cap2,
                            eff_in = 1,
                            eff_out = eta2,
                            p_cost = stor2.p_cost,
                            e_cost = stor2.e_cost)

    prod_wind_res = Production(power_ts = TimeSeries(np.array(power_res_new)
                    - prod_pv.power.data[:n], dt), p_cost= prod_wind.p_cost)

    os_res = OpSchedule(production_list = [prod_wind_res, prod_pv],
                storage_list = [stor1_res, stor2_res],
                production_p = [TimeSeries(prod_wind_res.power.data[:n], dt),
                                TimeSeries(prod_pv.power.data[:n], dt)],
                storage_p = [TimeSeries(p_vec1, dt),
                             TimeSeries(p_vec2, dt)],
                storage_e = [TimeSeries(e_vec1[:n], dt),
                             TimeSeries(e_vec2[:n], dt)],
                price = price_ts.data[:n])

    os_res.get_npv_irr(discount_rate, n_year)

    os_res.losses = [np.array(power_losses_bat) ,  np.array(power_losses_h2)]

    return os_res
