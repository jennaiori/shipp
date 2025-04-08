"""
This module defines kernel functions for shipp.

The functions defined in this module are used to analyze or compute data
for the classes defined in shipp.

Functions:
    solve_lp_pyomo: Build and solve a LP for NPV maximization with pyomo.
    solve_lp_alt_pyomo: Build and solve an alternative LP for NPV maximization with pyomo.
    solve_milp_pyomo: Build and solve a MILP for NPV maximization with pyomo.

"""

import numpy as np
import numpy_financial as npf

import pyomo.environ as pyo

from shipp.components import Storage, OpSchedule, Production
from shipp.timeseries import TimeSeries
from shipp.kernel import os_rule_based

def solve_lp_pyomo(price_ts: TimeSeries, prod_wind: Production,
                    prod_pv: Production, stor1: Storage, stor2: Storage,
                    discount_rate: float, n_year: int,
                    p_min, p_max: float,
                    n: int, name_solver: str = 'mosek', 
                    fixed_cap: bool = False) -> OpSchedule:
    """Build and solve a LP for NPV maximization with pyomo.

    This function builds and solves the hybrid sizing and operation problem as a linear program. 
    The objective is to minimize the Net Present Value of the plant. 
    In this function, the input for the power production represented by two Production objects, one for wind and one for solar.

    Params:
        price_ts (TimeSeries): Time series of the price of electricity on the day-ahead market [currency/MWh].
        prod_wind (Production): Object representing the power production from wind energy system.
        prod_pv (Production): Object representing the power production from solar PV system.
        stor1 (Storage): Object describing the battery storage.
        stor2 (Storage): Object describing the hydrogen storage system.
        discount_rate (float): Discount rate for the NPV calculation [-].
        n_year (int): Number of years for the NPV calculation [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.
        name_solver (str): Name of optimization solver to be used with pyomo.
        fixed_cap (bool): If True, the capacity of the storage is fixed.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time series do not match, if the length of the power in the Production objects is below n.
        RuntimeError: if the optimization algorithm fails to solve the problem.
    """

    # Data initialization
    dt = prod_wind.power.dt

    assert dt == price_ts.dt
    assert dt == prod_pv.power.dt
    assert n <=  len(prod_wind.power.data)
    assert n <=  len(prod_pv.power.data)
    assert n <=  len(price_ts.data)

    power_res = prod_wind.power.data[:n] + prod_pv.power.data[:n]

    p_cost1 = stor1.p_cost
    e_cost1 = stor1.e_cost
    eta1_in = stor1.eff_in
    eta1_out = stor1.eff_out

    p_cost2 = stor2.p_cost
    e_cost2 = stor2.e_cost
    eta2_in = stor2.eff_in
    eta2_out = stor2.eff_out

    # Check validity of input data
    assert np.all(np.isfinite(power_res))
    assert np.all(np.isfinite(p_min))
    assert np.isfinite(p_max)
    assert np.isfinite(dt)
    assert np.isfinite(eta1_in)
    assert np.isfinite(eta2_in)
    assert np.isfinite(eta1_out)
    assert np.isfinite(eta2_out)

    if isinstance(p_min, (np.ndarray, list)):
        assert len(p_min) >= n
        p_min_vec = p_min[:n].reshape(n,)
    elif isinstance(p_min, (float, int)):
        p_min_vec = p_min * np.ones((n,))
    else:
        raise ValueError("Input p_min in solve_lp_pyomo must be a float, int,\
                          list or numpy.array")


    # Build Concrete Model in Pyomo
    model = pyo.ConcreteModel()
    
    # Design Variables
    model.vec_n = pyo.Set(initialize=list(range(n)))
    model.vec_np1 = pyo.Set(initialize=list(range(n+1)))

    model.p_vec1 = pyo.Var(model.vec_n)
    model.e_vec1 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals)
    model.p_vec2 = pyo.Var(model.vec_n)
    model.e_vec2 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals)

    if stor1.p_cap == -1 or stor1.p_cap is None:
        model.p_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.p_cap1 = pyo.Var(bounds = (stor1.p_cap, stor1.p_cap))
    else:
        model.p_cap1 = pyo.Var(bounds = (0, stor1.p_cap))

    if stor2.p_cap == -1 or stor2.p_cap is None:
        model.p_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.p_cap2 = pyo.Var(bounds = (stor2.p_cap, stor2.p_cap))
    else:
        model.p_cap2 = pyo.Var(bounds = (0, stor2.p_cap))

    if stor1.e_cap == -1 or stor1.e_cap is None:
        model.e_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.e_cap1 = pyo.Var(bounds = (stor1.e_cap, stor1.e_cap))
    else:
        model.e_cap1 = pyo.Var(bounds = (0, stor1.e_cap))

    if stor2.e_cap == -1 or stor2.e_cap is None:
        model.e_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.e_cap2 = pyo.Var(bounds = (stor2.e_cap, stor2.e_cap))
    else:
        model.e_cap2 = pyo.Var(bounds = (0, stor2.e_cap))

    # Objective function
    factor = npf.npv(discount_rate, np.ones(n_year))-1
    model.obj = pyo.Objective(expr=  365 * 24/n*factor* sum([p*(model.p_vec1[n]
                 + model.p_vec2[k]) for p, n, k in zip(price_ts.data[:n],
                model.p_vec1, model.p_vec2)]) - p_cost1*model.p_cap1
                - e_cost1*model.e_cap1 - p_cost2*model.p_cap2
                - e_cost2*model.e_cap2, sense = pyo.maximize)

    # Rule functions for the constraints
    def rule_e_model_charge1(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] \
            <= - dt * eta1_in * model.p_vec1[i]

    def rule_e_model_discharge1(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] \
            <= - dt/eta1_out * model.p_vec1[i]

    def rule_p_max1(model, i):
        return model.p_vec1[i] <= model.p_cap1

    def rule_p_min1(model, i):
        return model.p_vec1[i] >= -model.p_cap1

    def rule_e_max1(model, i):
        return model.e_vec1[i] <= model.e_cap1

    def rule_e_model_charge2(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] \
            <= - dt * eta2_in * model.p_vec2[i]

    def rule_e_model_discharge2(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] \
            <= - dt/eta2_out * model.p_vec2[i]

    def rule_p_max2(model, i):
        return model.p_vec2[i] <= model.p_cap2

    def rule_p_min2(model, i):
        return model.p_vec2[i] >= -model.p_cap2

    def rule_e_max2(model, i):
        return model.e_vec2[i] <= model.e_cap2

    def rule_p_tot_min(model, i):
        return model.p_vec1[i] + model.p_vec2[i] >= p_min_vec[i]- power_res[i]

    def rule_p_tot_max(model, i):
        return model.p_vec1[i] + model.p_vec2[i] <= max(p_max - power_res[i],0)

    # Constraint for each storage type
    model.e_start_end1 =pyo.Constraint(expr = model.e_vec1[0]==model.e_vec1[n])
    model.e_start_end2 =pyo.Constraint(expr = model.e_vec2[0]==model.e_vec2[n])

    model.e_model_charge1 = pyo.Constraint(model.vec_n, rule=rule_e_model_charge1)
    model.e_model_discharge1 = pyo.Constraint(model.vec_n, rule=rule_e_model_discharge1)

    model.p_min1 = pyo.Constraint(model.vec_n, rule=rule_p_min1)
    model.p_max1 = pyo.Constraint(model.vec_n, rule=rule_p_max1)
    model.e_max1 = pyo.Constraint(model.vec_n, rule=rule_e_max1)

    model.e_model_charge2 = pyo.Constraint(model.vec_n, rule=rule_e_model_charge2)
    model.e_model_discharge2 = pyo.Constraint(model.vec_n, rule=rule_e_model_discharge2)

    model.p_min2 = pyo.Constraint(model.vec_n, rule=rule_p_min2)
    model.p_max2 = pyo.Constraint(model.vec_n, rule=rule_p_max2)
    model.e_max2 = pyo.Constraint(model.vec_n, rule=rule_e_max2)

    # Other constraints
    model.p_tot_min = pyo.Constraint(model.vec_n, rule=rule_p_tot_min)
    model.p_tot_max = pyo.Constraint(model.vec_n, rule=rule_p_tot_max)

    # Solve problem
    results = pyo.SolverFactory(name_solver).solve(model)
    # model.display()

    #Check if the problem was solved correclty
    if (results.solver.status is not pyo.SolverStatus.ok) or \
        (results.solver.termination_condition is not
         pyo.TerminationCondition.optimal):
        raise RuntimeError

    # Extract optimum
    e_vec1 = [pyo.value(model.e_vec1[e]) for e in model.e_vec1]
    p_vec1 = [pyo.value(model.p_vec1[e]) for e in model.p_vec1]
    e_cap1 = pyo.value(model.e_cap1)
    p_cap1 = pyo.value(model.p_cap1)

    e_vec2 = [pyo.value(model.e_vec2[e]) for e in model.e_vec2]
    p_vec2 = [pyo.value(model.p_vec2[e]) for e in model.p_vec2]
    e_cap2 = pyo.value(model.e_cap2)
    p_cap2 = pyo.value(model.p_cap2)


    # Calculate power losses
    power_res_new = []
    power_losses_bat = []
    power_losses_h2 = []
    for i in range(n):
        power_res_new.append(min(p_max - p_vec1[i] - p_vec2[i],
                             power_res[i]))

        power_losses_bat.append(-(e_vec1[i+1] - e_vec1[i] + dt*p_vec1[i])/dt)
        power_losses_h2.append(-(e_vec2[i+1] - e_vec2[i] + dt*p_vec2[i])/dt)

    # Build Storage, Production and OpSchedule objects for the optimum
    stor1_res = Storage(e_cap = e_cap1,
                        p_cap = p_cap1,
                        eff_in = eta1_in,
                        eff_out = eta1_out,
                        p_cost = stor1.p_cost,
                        e_cost = stor1.e_cost)
    stor2_res = Storage(e_cap = e_cap2,
                        p_cap = p_cap2,
                        eff_in = eta2_in,
                        eff_out = eta2_out,
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

    # Check that the power losses match the storage model
    if not os_res.check_losses(1e-7):
        print('Failed error check')
        os_res.check_losses(1e-7, True)
        raise RuntimeError

    # Extract solve time
    if name_solver in ('mosek', 'mosek_direct', 'cplex_direct'):
        os_res.time = results.solver[0]['Wallclock time']
    elif name_solver == 'gurobi':
        os_res.time = float(results.solver[0]['Wall time'])
    elif name_solver == 'cplex':
        os_res.time = results.solver[0]['Time']

    os_res.results = results
    
    return os_res


def solve_lp_alt_pyomo(price_ts: TimeSeries, prod_wind: Production,
                    prod_pv: Production, stor1: Storage, stor2: Storage,
                    discount_rate: float, n_year: int,
                    p_min, p_max: float,
                    n: int, name_solver: str = 'mosek',
                    fixed_cap: bool = False) -> OpSchedule:
    """Build and solve a LP for NPV maximization with pyomo.

    This function builds and solves the hybrid sizing and operation problem as a linear program. 
    The objective is to minimize the Net Present Value of the plant. 
    In this function, the input for the power production represented by two Production objects, one for wind and one for solar. 
    This is an alternative formulation, where the power from the storage is represented by two variables (for charge and discharge).
 
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
        name_solver (str): Name of optimization solver to be used with pyomo.
        fixed_cap (bool): If True, the capacity of the storage is fixed.

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
    eta1_in = stor1.eff_in
    eta1_out = stor1.eff_out

    p_cost2 = stor2.p_cost
    e_cost2 = stor2.e_cost
    eta2_in = stor2.eff_in
    eta2_out = stor2.eff_out


    assert np.all(np.isfinite(power_res))
    assert np.all(np.isfinite(p_min))
    assert np.isfinite(p_max)
    assert np.isfinite(dt)
    assert np.isfinite(eta1_in)
    assert np.isfinite(eta2_in)
    assert np.isfinite(eta1_out)
    assert np.isfinite(eta2_out)

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

    model.p_vec1_charge = pyo.Var(model.vec_n, domain = pyo.NonNegativeReals)
    model.p_vec1_discharge = pyo.Var(model.vec_n,domain = pyo.NonNegativeReals)
    model.e_vec1 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals)

    model.p_vec2_charge = pyo.Var(model.vec_n, domain = pyo.NonNegativeReals)
    model.p_vec2_discharge = pyo.Var(model.vec_n,domain = pyo.NonNegativeReals)
    model.p_vec2 = pyo.Var(model.vec_n)
    model.e_vec2 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals)

    if stor1.p_cap == -1 or stor1.p_cap is None:
        model.p_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.p_cap1 = pyo.Var(bounds = (stor1.p_cap, stor1.p_cap))
    else:
        model.p_cap1 = pyo.Var(bounds = (0, stor1.p_cap))

    if stor2.p_cap == -1 or stor2.p_cap is None:
        model.p_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.p_cap2 = pyo.Var(bounds = (stor2.p_cap, stor2.p_cap))
    else:
        model.p_cap2 = pyo.Var(bounds = (0, stor2.p_cap))

    if stor1.e_cap == -1 or stor1.e_cap is None:
        model.e_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.e_cap1 = pyo.Var(bounds = (stor1.e_cap, stor1.e_cap))
    else:
        model.e_cap1 = pyo.Var(bounds = (0, stor1.e_cap))

    if stor2.e_cap == -1 or stor2.e_cap is None:
        model.e_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.e_cap2 = pyo.Var(bounds = (stor2.e_cap, stor2.e_cap))
    else:
        model.e_cap2 = pyo.Var(bounds = (0, stor2.e_cap))

    #Objective
    factor = npf.npv(discount_rate, np.ones(n_year))-1
    model.obj = pyo.Objective(expr=  365 * 24/n*factor*
                              sum([p*(-model.p_vec1_charge[n1]
                 + model.p_vec1_discharge[n2] - model.p_vec2_charge[k1]
                 + model.p_vec2_discharge[k2]) for p, n1, n2, k1, k2 in \
                 zip(price_ts.data[:n], model.p_vec1_charge,
                      model.p_vec1_discharge, model.p_vec2_charge,
                      model.p_vec2_discharge)]) - p_cost1*model.p_cap1
                - e_cost1*model.e_cap1 - p_cost2*model.p_cap2
                - e_cost2*model.e_cap2, sense = pyo.maximize)

    # Rule functions for the constraints
    def rule_e_model1(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] \
            - dt*eta1_in*model.p_vec1_charge[i] \
            + dt/eta1_out*model.p_vec1_discharge[i] == 0

    def rule_p_max1(model, i):
        return model.p_vec1_discharge[i] <= model.p_cap1

    def rule_p_min1(model, i):
        return model.p_vec1_charge[i] <= model.p_cap1

    def rule_e_max1(model, i):
        return model.e_vec1[i] <= model.e_cap1

    def rule_e_model2(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] \
            - dt*eta2_in*model.p_vec2_charge[i] \
            + dt/eta2_out*model.p_vec2_discharge[i] == 0

    def rule_p_max2(model, i):
        return model.p_vec2_charge[i] <= model.p_cap2

    def rule_p_min2(model, i):
        return model.p_vec2_discharge[i] <= model.p_cap2

    def rule_e_max2(model, i):
        return model.e_vec2[i] <= model.e_cap2

    # def rule_p_tot_min(model, i):
    #   return model.p_vec1[i] + model.p_vec2[i] >= p_min_vec[i] - power_res[i]

    def rule_p_tot_min(model, i):
        return - model.p_vec1_charge[i] + model.p_vec1_discharge[i] \
               - model.p_vec2_charge[i] + model.p_vec2_discharge[i] \
               >= p_min_vec[i]- power_res[i]





    def rule_p_tot_max(model, i):
        return - model.p_vec1_charge[i] + model.p_vec1_discharge[i] \
               - model.p_vec2_charge[i] + model.p_vec2_discharge[i] \
               <= max(p_max - power_res[i], 0)

    # Constraint for each storage type
    model.e_start_end1 =pyo.Constraint(expr = model.e_vec1[0]==model.e_vec1[n])
    model.e_start_end2 =pyo.Constraint(expr = model.e_vec2[0]==model.e_vec2[n])

    model.e_model1 = pyo.Constraint(model.vec_n,
                                           rule=rule_e_model1)

    model.p_min1 = pyo.Constraint(model.vec_n, rule=rule_p_min1)
    model.p_max1 = pyo.Constraint(model.vec_n, rule=rule_p_max1)
    model.e_max1 = pyo.Constraint(model.vec_n, rule=rule_e_max1)

    model.e_model2 = pyo.Constraint(model.vec_n,
                                           rule=rule_e_model2)

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
    p_vec1 = [-pyo.value(model.p_vec1_charge[e1])
              + pyo.value(model.p_vec1_discharge[e2])
              for e1, e2 in zip(model.p_vec1_charge, model.p_vec1_discharge)]
    e_cap1 = pyo.value(model.e_cap1)
    p_cap1 = pyo.value(model.p_cap1)

    e_vec2 = [pyo.value(model.e_vec2[e]) for e in model.e_vec2]
    p_vec2 = [-pyo.value(model.p_vec2_charge[e1])
              + pyo.value(model.p_vec2_discharge[e2])
              for e1, e2 in zip(model.p_vec2_charge, model.p_vec2_discharge)]
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
                            eff_in = eta1_in,
                            eff_out = eta1_out,
                            p_cost = stor1.p_cost,
                            e_cost = stor1.e_cost)
    stor2_res = Storage(e_cap = e_cap2,
                            p_cap = p_cap2,
                            eff_in = eta2_in,
                            eff_out = eta2_out,
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

    if not os_res.check_losses(1e-7):
        print('Failed error check')
        os_res.check_losses(1e-7, True)



        raise RuntimeError

    if name_solver in ('mosek', 'mosek_direct', 'cplex_direct'):
        os_res.time = results.solver[0]['Wallclock time']
    elif name_solver == 'gurobi':
        os_res.time = float(results.solver[0]['Wall time'])
    elif name_solver == 'cplex':
        os_res.time = results.solver[0]['Time']

    os_res.results = results

    return os_res


def solve_milp_pyomo(price_ts: TimeSeries, prod_wind: Production,
                    prod_pv: Production, stor1: Storage, stor2: Storage,
                    discount_rate: float, n_year: int,
                    p_min, p_max: float,
                    n: int, name_solver: str = 'mosek',
                    fixed_cap: bool = False) -> OpSchedule:
    """Build and solve a MILP for NPV maximization with pyomo.

    This function builds and solves the hybrid sizing and operation problem as a mixed-integer linear program. 
    The objective is to minimize the Net Present Value of the plant. 
    In this function, the input for the power production represented by two Production objects, one for wind and one for solar.

    Params:
        price_ts (TimeSeries): Time series of the price of electricity on the day-ahead market [currency/MWh].
        prod_wind (Production): Object representing the power production from wind energy system.
        prod_pv (Production): Object representing the power production from solar PV system.
        stor1 (Storage): Object describing the battery storage.
        stor2 (Storage): Object describing the hydrogen storage system.
        discount_rate (float): Discount rate for the NPV calculation [-].
        n_year (int): Number of years for the NPV calculation [-].
        p_min (float or np.ndarray): Minimum power requirement [MW].
        p_max (float): Maximum power requirement [MW].
        n (int): Number of time steps to consider in the optimization.
        name_solver (str): Name of optimization solver to be used with pyomo.
        fixed_cap (bool): If True, the capacity of the storage is fixed.

    Returns:
        os_res (OpSchedule): Object describing the optimal operational schedule and storage size.

    Raises:
        AssertionError: if the time step of the power and price time series do not match, if the length of the power in the Production objects is below n.
        RuntimeError: if the optimization algorithm fails to solve the problem.
    """
    dt = prod_wind.power.dt

    assert dt == price_ts.dt
    assert dt == prod_pv.power.dt
    assert n <=  len(prod_wind.power.data)
    assert n <=  len(prod_pv.power.data)
    assert n <=  len(price_ts.data)

    power_res = prod_wind.power.data[:n] + prod_pv.power.data[:n]

    p_cap1 = stor1.p_cap
    e_cap1 = stor1.e_cap
    p_cost1 = stor1.p_cost
    e_cost1 = stor1.e_cost
    eta1_in = stor1.eff_in
    eta1_out = stor1.eff_out

    p_cap2 = stor2.p_cap
    e_cap2 = stor2.e_cap
    p_cost2 = stor2.p_cost
    e_cost2 = stor2.e_cost
    eta2_in = stor2.eff_in
    eta2_out = stor2.eff_out


    assert np.all(np.isfinite(power_res))
    assert np.all(np.isfinite(p_min))
    assert np.isfinite(p_max)
    assert np.isfinite(dt)
    assert np.isfinite(eta1_in)
    assert np.isfinite(eta2_in)
    assert np.isfinite(eta1_out)
    assert np.isfinite(eta2_out)

    if isinstance(p_min, (np.ndarray, list)):
        assert len(p_min) >= n
        p_min_vec = p_min[:n].reshape(n,)
    elif isinstance(p_min, (float, int)):
        p_min_vec = p_min * np.ones((n,))
    else:
        raise ValueError("Input p_min in solve_lp_pyomo must be a float, int,\
                          list or numpy.array")

    bigM = 10 * p_max

    # Build Concrete Model in Pyomo
    model = pyo.ConcreteModel()

    # Design Variables
    model.vec_n = pyo.Set(initialize=list(range(n)))
    model.vec_np1 = pyo.Set(initialize=list(range(n+1)))

    model.p_vec1 = pyo.Var(model.vec_n, domain = pyo.Reals, bounds=(-p_cap1, p_cap1))
    model.e_vec1 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals, bounds = (0, e_cap1))
    model.bin1 = pyo.Var(model.vec_n, domain = pyo.Binary)

    model.p_vec2 = pyo.Var(model.vec_n, domain = pyo.Reals, bounds = (-p_cap2, p_cap2))
    model.e_vec2 = pyo.Var(model.vec_np1, domain = pyo.NonNegativeReals, bounds=(0, e_cap2))
    model.bin2 = pyo.Var(model.vec_n, domain = pyo.Binary)

    if stor1.p_cap == -1 or stor1.p_cap is None:
        model.p_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.p_cap1 = pyo.Var(bounds = (stor1.p_cap, stor1.p_cap))
    else:
        model.p_cap1 = pyo.Var(bounds = (0, stor1.p_cap))

    if stor2.p_cap == -1 or stor2.p_cap is None:
        model.p_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.p_cap2 = pyo.Var(bounds = (stor2.p_cap, stor2.p_cap))
    else:
        model.p_cap2 = pyo.Var(bounds = (0, stor2.p_cap))

    if stor1.e_cap == -1 or stor1.e_cap is None:
        model.e_cap1 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.e_cap1 = pyo.Var(bounds = (stor1.e_cap, stor1.e_cap))
    else:
        model.e_cap1 = pyo.Var(bounds = (0, stor1.e_cap))

    if stor2.e_cap == -1 or stor2.e_cap is None:
        model.e_cap2 = pyo.Var(domain = pyo.NonNegativeReals)
    elif fixed_cap == True:
        model.e_cap2 = pyo.Var(bounds = (stor2.e_cap, stor2.e_cap))
    else:
        model.e_cap2 = pyo.Var(bounds = (0, stor2.e_cap))

    # Objective function
    factor = npf.npv(discount_rate, np.ones(n_year))-1
    model.obj = pyo.Objective(expr=  365 * 24/n*factor* sum([p*(model.p_vec1[n]
                 + model.p_vec2[k]) for p, n, k in zip(price_ts.data[:n],
                model.p_vec1, model.p_vec2)]) - p_cost1*model.p_cap1
                - e_cost1*model.e_cap1 - p_cost2*model.p_cap2
                - e_cost2*model.e_cap2, sense = pyo.maximize)


    # Rule functions for the constraints
    def rule_e_model_charge1_lb(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] + dt * eta1_in * \
            model.p_vec1[i] >= -bigM * model.bin1[i]

    def rule_e_model_charge1_ub(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] + dt * eta1_in * \
            model.p_vec1[i] <= bigM * model.bin1[i]

    def rule_e_model_discharge1_lb(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] + dt/eta1_out * \
            model.p_vec1[i] >= - bigM*(1 - model.bin1[i])

    def rule_e_model_discharge1_ub(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] + dt/eta1_out * \
            model.p_vec1[i] <=  bigM*(1 - model.bin1[i])

    def rule_p_max1(model, i):
        return model.p_vec1[i] <= model.p_cap1

    def rule_p_min1(model, i):
        return model.p_vec1[i] >= -model.p_cap1

    def rule_e_max1(model, i):
        return model.e_vec1[i] <= model.e_cap1

    def rule_p_bin1_lb(model, i):
        return  -bigM *(1 - model.bin1[i]) <= model.p_vec1[i]

    def rule_p_bin1_ub(model, i):
        return   model.p_vec1[i] <= bigM * model.bin1[i]

    ### --- --- ---

    def rule_e_model_charge2_lb(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] + dt * eta2_in * \
            model.p_vec2[i] >= -bigM * model.bin2[i]

    def rule_e_model_charge2_ub(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] + dt * eta2_in * \
            model.p_vec2[i] <= bigM * model.bin2[i]

    def rule_e_model_discharge2_lb(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] + dt/eta2_out * \
            model.p_vec2[i] >= - bigM*(1 - model.bin2[i])

    def rule_e_model_discharge2_ub(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] + dt/eta2_out * \
            model.p_vec2[i] <=  bigM*(1 - model.bin2[i])

    def rule_p_max2(model, i):
        return model.p_vec2[i] <= model.p_cap2

    def rule_p_min2(model, i):
        return model.p_vec2[i] >= -model.p_cap2

    def rule_e_max2(model, i):
        return model.e_vec2[i] <= model.e_cap2

    def rule_p_bin2_lb(model, i):
        return  -bigM *(1 - model.bin2[i]) <= model.p_vec2[i]

    def rule_p_bin2_ub(model, i):
        return   model.p_vec2[i] <= bigM * model.bin2[i]

    def rule_p_tot1(model, i):
        return model.e_vec1[i+1]-model.e_vec1[i] + dt * model.p_vec1[i] <= 0

    def rule_p_tot2(model, i):
        return model.e_vec2[i+1]-model.e_vec2[i] + dt * model.p_vec2[i] <= 0

    def rule_p_tot_min(model, i):
        return model.p_vec1[i] + model.p_vec2[i] >= p_min_vec[i] - power_res[i]

    def rule_p_tot_max(model, i):
        return model.p_vec1[i] + model.p_vec2[i] <= max(p_max - power_res[i], 0)

    # Constraint for each storage type
    model.e_start_end1 =  pyo.Constraint(expr = model.e_vec1[0] == model.e_vec1[n])
    model.e_start_end2 =  pyo.Constraint(expr = model.e_vec2[0] == model.e_vec2[n])

    model.e_model_charge1_ub = pyo.Constraint(model.vec_n, rule=rule_e_model_charge1_ub)
    model.e_model_charge1_lb = pyo.Constraint(model.vec_n,
                                              rule=rule_e_model_charge1_lb)
    model.e_model_discharge1_ub = pyo.Constraint(model.vec_n,
                                            rule=rule_e_model_discharge1_ub)
    model.e_model_discharge1_lb = pyo.Constraint(model.vec_n,
                                            rule=rule_e_model_discharge1_lb)
    model.p_tot1 = pyo.Constraint(model.vec_n, rule = rule_p_tot1)

    model.p_bin1_ub = pyo.Constraint(model.vec_n, rule=rule_p_bin1_ub)
    model.p_bin1_lb = pyo.Constraint(model.vec_n, rule=rule_p_bin1_lb)

    model.p_min1 = pyo.Constraint(model.vec_n, rule=rule_p_min1)
    model.p_max1 = pyo.Constraint(model.vec_n, rule=rule_p_max1)
    model.e_max1 = pyo.Constraint(model.vec_n, rule=rule_e_max1)

    model.e_model_charge2_ub = pyo.Constraint(model.vec_n,
                                            rule=rule_e_model_charge2_ub)
    model.e_model_charge2_lb = pyo.Constraint(model.vec_n,
                                            rule=rule_e_model_charge2_lb)
    model.e_model_discharge2_ub = pyo.Constraint(model.vec_n,
                                            rule=rule_e_model_discharge2_ub)
    model.e_model_discharge2_lb = pyo.Constraint(model.vec_n,
                                            rule=rule_e_model_discharge2_lb)
    model.p_tot2 = pyo.Constraint(model.vec_n, rule = rule_p_tot2)

    model.p_bin2_ub = pyo.Constraint(model.vec_n, rule=rule_p_bin2_ub)
    model.p_bin2_lb = pyo.Constraint(model.vec_n, rule=rule_p_bin2_lb)

    model.p_min2 = pyo.Constraint(model.vec_n, rule=rule_p_min2)
    model.p_max2 = pyo.Constraint(model.vec_n, rule=rule_p_max2)
    model.e_max2 = pyo.Constraint(model.vec_n, rule=rule_e_max2)

    # General constraints
    model.p_tot_min = pyo.Constraint(model.vec_n, rule=rule_p_tot_min)
    model.p_tot_max = pyo.Constraint(model.vec_n, rule=rule_p_tot_max)

    # Solve optimization problem
    results = pyo.SolverFactory(name_solver).solve(model)
    # model.display()

    #Check if the problem was solved correclty
    if (results.solver.status is not pyo.SolverStatus.ok) or \
        (results.solver.termination_condition is not
         pyo.TerminationCondition.optimal):
        raise RuntimeError

    # Extract optimum
    e_vec1 = [pyo.value(model.e_vec1[e]) for e in model.e_vec1]
    p_vec1 = [pyo.value(model.p_vec1[e]) for e in model.p_vec1]
    e_cap1 = pyo.value(model.e_cap1)
    p_cap1 = pyo.value(model.p_cap1)

    e_vec2 = [pyo.value(model.e_vec2[e]) for e in model.e_vec2]
    p_vec2 = [pyo.value(model.p_vec2[e]) for e in model.p_vec2]
    e_cap2 = pyo.value(model.e_cap2)
    p_cap2 = pyo.value(model.p_cap2)

    # Calculate losses
    power_res_new = []
    power_losses_bat = []
    power_losses_h2 = []
    for i in range(n):
        power_res_new.append(min(p_max - p_vec1[i] - p_vec2[i],
                             power_res[i]))

        power_losses_bat.append(-(e_vec1[i+1] - e_vec1[i] + dt*p_vec1[i])/dt)
        power_losses_h2.append(-(e_vec2[i+1] - e_vec2[i] + dt*p_vec2[i])/dt)

    # Create Storage, Production and OpSchedule objects for the optimum
    stor1_res = Storage(e_cap = e_cap1,
                            p_cap = p_cap1,
                            eff_in = eta1_in,
                            eff_out = eta1_out,
                            p_cost = stor1.p_cost,
                            e_cost = stor1.e_cost)
    stor2_res = Storage(e_cap = e_cap2,
                            p_cap = p_cap2,
                            eff_in = eta2_in,
                            eff_out = eta2_out,
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

    os_res.losses = [np.array(power_losses_bat), np.array(power_losses_h2)]

    # Extract solve time
    if name_solver in ('mosek', 'mosek_direct', 'cplex_direct'):
        os_res.time = results.solver[0]['Wallclock time']
    elif name_solver == 'gurobi':
        os_res.time = float(results.solver[0]['Wall time'])
    elif name_solver == 'cplex':
        os_res.time = results.solver[0]['Time']

    os_res.results = results

    return os_res


def run_storage_operation(run_type: str, power: list, price: list, p_min: float, p_max: float, stor: Storage, e_start: float, n: int, nt: int, dt: float, rel : float = 1.0, forecast: list = None, n_hist: int = 0, verbose : bool = False, name_solver : str = 'mosek') -> dict:	
    """Execute a storage operation simulation.
    
    This function simulates the dispatch operation of a storage system based on the specified type. 
    It supports rule-based operation, and dispatch optimization based on point and ensemble forecast
    as well as unlimited information to determine the power and energy levels of the storage system 
    over a given time horizon.

    Params:
        run_type (str): The type of operation to run. Options are 'unlimited', 'rule-based', or 'forecast'.
            - 'unlimited': Solves a unlimited optimization problem to maximize revenues.
            - 'rule-based': Uses predefined rules for storage operation based on thresholds for power and price.
            - 'forecast': Considers a rolling horizon with a new power forecast at each time step.
        power (list): A list of power values (e.g., renewable generation) over the time horizon.
        price (list): A list of price values over the time horizon.
        p_min (float): Minimum power threshold for the storage operation.
        p_max (float): Maximum power threshold for the storage operation.
        stor (Storage): The storage object containing storage parameters (e.g., capacity, efficiency).
        e_start (float): Initial energy level in the storage.
        n (int): Number of time steps for the forecast.
        nt (int): Number of time steps in the simulation.
        dt (float): Time step duration in hours.
        rel (float, optional): Reliability threshold for the operation. Default is 1.0.
        forecast (list, optional): Forecasted power scenarios for the 'forecast' run type. Default is None.
        n_hist (int, optional): Number of historical time steps to consider for reliability in the 'forecast' run type. Default is 0.
        verbose (bool, optional): If True, enables verbose output during the simulation. Default is False.
        name_solver (str, optional): Name of the solver to use for optimization. Default is 'mosek'.

    Returns:
        dict: A dictionary containing the results of the storage operation simulation:
            - 'power' (list): Power output of the storage over the time horizon.
            - 'energy' (list): Energy levels of the storage over the time horizon.
            - 'reliability' (float): Reliability of the operation (fraction of time steps meeting the power threshold).
            - 'revenues' (float): Total revenues from the storage operation.
            - 'cost' (float): Cost of the operation per unit of power.
            - 'bin' (list): Binary indicators for whether the power threshold was met at each time step.

    Raises:
        RuntimeError: If an invalid `run_type` is provided.
        AssertionError: If `forecast` is None when `run_type` is 'forecast'.
    """
    # Validate input data
    assert isinstance(run_type, str), "run_type must be a string."
    assert run_type in ['unlimited', 'rule-based', 'forecast'], "Invalid run_type. Must be 'unlimited', 'rule-based', or 'forecast'."
    assert isinstance(p_min, (int, float)) and p_min >= 0, "p_min must be a non-negative number."
    assert isinstance(p_max, (int, float)) and p_max > p_min, "p_max must be greater than p_min."
    assert isinstance(stor, Storage), "stor must be an instance of the Storage class."
    assert isinstance(e_start, (int, float)) and 0 <= e_start <= stor.e_cap, "e_start must be within the storage capacity."
    assert isinstance(n, int) and n > 0, "n must be a positive integer."
    assert isinstance(nt, int) and nt > 0, "nt must be a positive integer."
    assert isinstance(dt, (int, float)) and dt > 0, "dt must be a positive number."
    assert isinstance(rel, (int, float)) and 0 <= rel <= 1, "rel must be a number between 0 and 1."
    if run_type == 'forecast':
        assert forecast is not None, "forecast must be provided for 'forecast' run_type."
        assert isinstance(forecast, list) and all(isinstance(f, list) for f in forecast), "forecast must be a list of lists."
        assert all(len(f[0]) >= n for f in forecast), "Each forecast scenario must have at least n time steps."
    assert isinstance(n_hist, int) and n_hist >= 0, "n_hist must be a non-negative integer."
    assert isinstance(verbose, bool), "verbose must be a boolean."
    assert isinstance(name_solver, str), "name_solver must be a string."

    # Initialize parameters for the simulation
    stor_null = Storage(e_cap = 0,  p_cap = 0, eff_in = 1.0, eff_out = 1.0)
    tol = 1e-6
    
    # For run type 'unlimited', the information for the entire time series is used, and there is no rolling horizon.
    if run_type == 'unlimited':
        power = np.array(power)
        p_vec, e_vec, _, _, bin_res, _ = solve_dispatch_pyomo(price, 1, rel, nt, power.reshape((1,len(power))), p_min, p_max, e_start, 0,  dt, stor, stor_null, verbose = verbose, name_solver = name_solver)
        
        p_res = p_vec[0].tolist()
        e_res = e_vec[0].tolist()

    # For run type 'rule-based', the operation is based on predefined rules and thresholds. The best thresholds are determined by iterating over a range of values.
    elif run_type == 'rule-based':
        # The power and price data are reshaped as TimeSeries objects.
        price_ts = TimeSeries(price[:nt], dt)
        prod_wind = Production(TimeSeries(power[:nt], dt), 0)
        prod_pv = Production(TimeSeries([0 for _ in range(nt)], dt), 0)

        n_rb = 20 # Number of values to iterate over for the threshold parameters.


        p_rule_vec = np.linspace(p_min, p_max-1, n_rb)
        price_min_vec = np.linspace(np.mean(price[:nt]), max(price[:nt]), n_rb)

        max_rev = -max(price)*p_max*nt*dt # Initializing maximum revenue to a very low value.
        os = os_rule_based(price_ts, prod_wind, prod_pv, stor, stor_null, 
                    0.05, 10, p_min, p_min, 0, nt, e_start)
        for p_rule in p_rule_vec:
            for price_min in price_min_vec:
                os_tmp = os_rule_based(price_ts, prod_wind, prod_pv, stor, stor_null, 
                    0.05, 10, p_min, p_rule, price_min, nt, e_start)
                
                p_res = os_tmp.storage_p[0].data.tolist()
                rel_res = sum([1/nt if (p + ps) >= p_min-tol else 0 for ps, p in zip(p_res[:nt], power[:nt])]) # calculate reliability
                rev_res = sum([price[i]*p_res[i] for i in range(nt)]) # calculate revenues

                if rel_res >= rel and max_rev < rev_res:
                    max_rev = max(rev_res, max_rev)
                    os = os_tmp
                elif max_rev < rev_res:
                    max_rev = max(rev_res, max_rev)
                    os = os_tmp                          

        # Retrieve results for the best case found.
        p_res = os.storage_p[0].data.tolist()
        e_res = os.storage_e[0].data.tolist()
        bin_res = [p + pwr>=p_min for p, pwr in zip(p_res, power)]
  
    # For run type 'forecast', the operation is based on a rolling horizon with new power forecasts at each time step.	
    elif run_type == 'forecast':
        assert forecast is not None
    
        p_res = []
        e_res = [e_start]
        bin_res = []
        e_start_new = e_start
        
        m = len(forecast[0]) # Number of forecast scenarios.

        cnt_hist = n_hist # Initializing the count of historical time steps meeting the power threshold.

        # Iterate over the time steps in the simulation for the rolling horizon.
        for t in range(nt):
            # If the current time step is lower than the length of the time window for past operation, we use a smaller time window for the optimization.
            if t > n_hist:
                cnt_hist = sum([0 if p+ps < p_min else 1 for ps, p in zip(p_res[-n_hist:], power[t-n_hist:t])])

                p_vec, e_vec, _, _, bin_vec, status = solve_dispatch_pyomo(price[t:], m, rel, n, forecast[t], p_min, p_max, e_start_new, 0,  dt, stor, stor_null, n_hist = n_hist, cnt_hist=cnt_hist, verbose = verbose, name_solver = name_solver)
            else: 
                cnt_hist = sum([0 if p+ps < p_min else 1 for ps, p in zip(p_res[:t], power[:t])])
        
                p_vec, e_vec, _, _, bin_vec, status = solve_dispatch_pyomo(price[t:], m, rel, n, forecast[t], p_min, p_max, e_start_new, 0,  dt, stor, stor_null, n_hist = n_hist, cnt_hist=(t-1), verbose = verbose, name_solver = name_solver)
            
            # If the optimization problem is solved correctly, we retrieve the results.
            if status == 'ok':
                e_start_new = e_vec[0,1]
                p_new = p_vec[0,0]
            else:
                print('Time step warning:', t, t/24, e_start_new)
                # If the optimization solver fails, calculate the power and energy levels so that the storage system charges or discharge to meet the baseload power level.
                delta_power = p_min - power[t]
                if delta_power >= 0:
                    p_new = min(e_res[t]*stor.eff_out/dt, delta_power, stor.p_cap)
                    e_start_new = e_res[t] - p_new*dt/stor.eff_out
                else:
                    p_new = max(-1.0/dt*(stor.e_cap-e_res[t]), delta_power, -stor.p_cap)
                    e_start_new = e_res[t] - p_new*dt

        
            e_res.append(e_start_new)
            p_res.append(p_new)
            bin_res.append(bin_vec[0])

    else:
        print('Incorrect run type:', run_type)
        raise(RuntimeError)
    
    # Calculate reliability, revenues and costs based on the results.
    rel_res = sum([1/nt if (p + ps) >= p_min-tol else 0 for ps, p in zip(p_res[:nt], power[:nt])])
    rev_res = sum([price[i]*p_res[i] for i in range(nt)])
    cost_res = -rev_res/(sum(power[:nt])*dt)
    
    res = {'power':p_res, 'energy': e_res, 'reliability': rel_res, 
           'revenues': rev_res, 'cost': cost_res, 
           'bin': np.array(bin_res).astype(int).tolist()}

    return res


def solve_dispatch_pyomo(price: list, m: int, rel: float, n: int, power_forecast: list, p_min: float, p_max : float, e_start1 : float, e_start2 : float, dt : float, stor1 : Storage, stor2 : Storage, cnt_hist : int = 0, n_hist : int = 0, verbose : bool = False, name_solver : str = 'mosek')-> tuple:

    """Build and solve a MILP for the dispatch optimization of storage systems

    This function builds and solves the dispatch optimization problem for two storage systems operating a minimum baseload power as a mixed integer linear program. 
    The objective is to maximize the income of the storage system (i.e. revenues on the spot market). 

    Params:
        price (list): Price of electricity on the day-ahead market [currency/MWh].
        m (int): Number of forecast scenarios.
        rel (float): Percentage of baseload reliability required.
        n (int): Number of time steps to consider in the optimization (i.e. forecast lead-time)
        power_forecast (list): Forecasted power scenarios for the optimization. 
        p_min (float): Minimum baseload power requirement [MW].
        p_max (float): Maximum power [MW].
        e_start1 (float): Initial energy level in the first storage system [MWh].
        e_start2 (float): Initial energy level in the second storage system [MWh].
        dt (float): Time step duration in hours.
        stor1 (Storage): Object describing the first storage system (e.g., battery storage).
        stor2 (Storage): Object describing the second storage system (e.g., hydrogen storage).
        cnt_hist (int, optional): Number of time steps meeting the power threshold within the time window for past operation. Default is 0.
        n_hist (int, optional): Number of time steps for the time window for past operation. Default is 0.
        verbose (bool, optional): If True, enables verbose output during the optimization. Default is False.
        name_solver (str, optional): Name of the optimization solver to be used with Pyomo. Default is 'mosek'.

    Returns:
        tuple: A tuple containing the following elements:
            - p_vec1 (np.ndarray): Power output of the first storage system over the time horizon.
            - e_vec1 (np.ndarray): Energy levels of the first storage system over the time horizon.
            - p_vec2 (np.ndarray): Power output of the second storage system over the time horizon.
            - e_vec2 (np.ndarray): Energy levels of the second storage system over the time horizon.
            - bin (np.ndarray): Binary indicators for whether the power threshold was met at each time step.
            - results.solver.status: Solver status indicating the outcome of the optimization.

    Raises:
        RuntimeError: If the optimization algorithm fails to solve the problem.
    """


    # Check validity of input data

    assert m <= len(power_forecast)
    assert n <=  len(power_forecast[0])
    assert n <=  len(price)
    assert min(price) >=0
    assert rel >=0 and rel <= 1.0

    assert np.all(np.isfinite(power_forecast))
    assert np.all(np.isfinite(price))
    assert np.isfinite(p_min)
    assert np.isfinite(p_max)
    assert np.isfinite(dt)
    
    #Load storage system parameters from storage objects
    p_cap1 = stor1.p_cap
    e_cap1 = stor1.e_cap
    eta1_in = stor1.eff_in
    eta1_out = stor1.eff_out

    p_cap2 = stor2.p_cap
    e_cap2 = stor2.e_cap
    eta2_in = stor2.eff_in
    eta2_out = stor2.eff_out

    assert np.isfinite(eta1_in)
    assert np.isfinite(eta2_in)
    assert np.isfinite(eta1_out)
    assert np.isfinite(eta2_out)
    assert np.isfinite(p_cap1)
    assert np.isfinite(p_cap2)
    assert np.isfinite(e_cap1)
    assert np.isfinite(e_cap2)

    # Tuning parameters of the optimization problem
    mu_obj = 1.0*p_min*(n+n_hist)*np.max(price[0:n])
    beta_obj = 1e-6

    # Initialize pyomo model
    model = pyo.ConcreteModel()

    # Initialize Sets for Design Variables
    model.vec_n = pyo.Set(initialize=list(range(n)))
    model.vec_m = pyo.Set(initialize=list(range(m)))
    model.vec_np1 = pyo.Set(initialize=list(range(n+1)))

    model.mat_m_n = pyo.Set(initialize=model.vec_m*model.vec_n)
    model.mat_m_np1 = pyo.Set(initialize=model.vec_m*model.vec_np1)

    # Initialize Design Variables
    model.p_vec1 = pyo.Var(model.mat_m_n, bounds = (-p_cap1, p_cap1))
    model.e_vec1 = pyo.Var(model.mat_m_np1, bounds = (0, e_cap1))

    model.p_vec2 = pyo.Var(model.mat_m_n, bounds = (-p_cap2, p_cap2))
    model.e_vec2 = pyo.Var(model.mat_m_np1, bounds = (0, e_cap2))

    model.bin = pyo.Var(model.vec_n, within = pyo.Binary, 
                        initialize = [0 if p < p_min else 1 for p in power_forecast[0][:n]]) ##binary variable for each time step

    model.penalty = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, rel), initialize=0)
   
    # Input the objective function in the model
    model.obj = pyo.Objective(expr = 1/m*sum([price[j] * (model.p_vec1[i,j] + model.p_vec2[i,j]) for i,j in model.p_vec1]) - mu_obj*model.penalty + beta_obj*1/m*sum([model.e_vec1[i, model.vec_np1.at(n+1)] + model.e_vec2[i, model.vec_np1.at(n+1)] for i in model.vec_m]), sense = pyo.maximize)


    # Define rule functions for the constraints
    ## Constraint for the minimum baseload power
    def rule_p_tot_min(model, j, i):
        return model.p_vec1[j,i] + model.p_vec2[j,i] >= (p_min*model.bin[i] 
                                     - power_forecast[j][i])
    ## Constraint for the maximum power
    def rule_p_tot_max(model, j, i):
        return model.p_vec1[j,i] + model.p_vec2[j,i] <= max(p_max - power_forecast[j][i], 0)

    ## Constraints for the storage system model for storage 1
    def rule_e_model_charge1(model, j,i):
        return model.e_vec1[j,i+1]-model.e_vec1[j,i] <= - dt * eta1_in * model.p_vec1[j, i]
    def rule_e_model_discharge1(model, j,i):
        return model.e_vec1[j,i+1]-model.e_vec1[j,i] <= - dt/eta1_out * model.p_vec1[j, i]

    ## Constraint for the initial energy level of storage 1
    def rule_e_equal1(model, j):
        # return model.e_vec1[0,0] == model.e_vec1[j,0]
        return model.e_vec1[j,0] == e_start1

    ## Constraint to ensure the first time step is equal for all scenarios, in terms of power output of storage 1
    def rule_p_equal1(model, j):
        return model.p_vec1[0,0] == model.p_vec1[j,0]
    
     ## Constraints for the storage system model for storage 2
    def rule_e_model_charge2(model, j,i):
        return model.e_vec2[j,i+1]-model.e_vec2[j,i] <= - dt * eta2_in * model.p_vec2[j, i]
    def rule_e_model_discharge2(model, j,i):
        return model.e_vec2[j,i+1]-model.e_vec2[j,i] <= - dt/eta2_out * model.p_vec2[j, i]

    ## Constraint for the initial energy level of storage 1
    def rule_e_equal2(model, j):
        # return model.e_vec2[0,0] == model.e_vec2[j,0]
        return model.e_vec2[j,0] == e_start2

    ## Constraint to ensure the first time step is equal for all scenarios, in terms of power output of storage 2
    def rule_p_equal2(model, j):
        return model.p_vec2[0,0] == model.p_vec2[j,0]
    
    ## Constraint for the binary variables to ensure that the baseload power is always satisfied if the power from RES sources is above baseload
    def rule_min_bound_bin(model, i):
        if p_min >0:
            min_pow = min([power_forecast[j][i] for j in range(m)])
            return model.bin[i] >= (int)((min_pow//p_min)>=1)
        else:
            return model.bin[i]>=1
   
    # Input the constraints in the model
    ## Constraint to enforce the target reliability, taking into account the penalty
    model.reliability = pyo.Constraint(expr = sum( model.bin[k] for k in model.bin) 
                                       >= (rel - model.penalty)* (n+n_hist) - cnt_hist)
    ## Constraint for the binary variables
    model.min_bound_bin = pyo.Constraint(model.vec_n, rule=rule_min_bound_bin)
    ## Constraint for each storage type
    model.e_model_charge1 = pyo.Constraint(model.mat_m_n, rule=rule_e_model_charge1)
    model.e_model_discharge1 = pyo.Constraint(model.mat_m_n, rule=rule_e_model_discharge1)
    model.e_model_charge2 = pyo.Constraint(model.mat_m_n, rule=rule_e_model_charge2)
    model.e_model_discharge2 = pyo.Constraint(model.mat_m_n, rule=rule_e_model_discharge2)

    ## Global constraints
    model.p_tot_min = pyo.Constraint(model.mat_m_n, rule=rule_p_tot_min)
    model.p_tot_max = pyo.Constraint(model.mat_m_n, rule=rule_p_tot_max)

    ## Link all three together
    model.equal_p1 = pyo.Constraint(model.vec_m, rule=rule_p_equal1)
    model.equal_e1 = pyo.Constraint(model.vec_m, rule=rule_e_equal1)
    model.equal_p2 = pyo.Constraint(model.vec_m, rule=rule_p_equal2)
    model.equal_e2 = pyo.Constraint(model.vec_m, rule=rule_e_equal2)
    
    # Define solver object
    opt = pyo.SolverFactory(name_solver)

    # Enfore limits on the solve time
    if n < 7*24:
        time_limit = 2
    else:
        time_limit = 3*60 # 3 minutes
    if 'cplex' in name_solver:
        opt.options['timelimit'] = time_limit
    elif 'gurobi' in name_solver:           
        opt.options['TimeLimit'] = time_limit
    elif 'mosek' in name_solver:
        opt.options['dparam.optimizer_max_time'] = time_limit 

    # Solve optimization problem
    if verbose:
        results = opt.solve(model, tee=True)
    else:
        results = opt.solve(model, tee=False)

    # Extract optimum for each design variable
    bin = np.zeros(n)
    for i in model.bin:
        bin[i] = pyo.value(model.bin[i])

    p_vec1 = np.zeros((m,n))
    for j, i in model.p_vec1:
        p_vec1[j][i] = pyo.value(model.p_vec1[j, i])

    e_vec1 = np.zeros((m,n+1))
    for j, i in model.e_vec1:
        e_vec1[j][i] = pyo.value(model.e_vec1[j, i])

    p_vec2 = np.zeros((m,n))
    for j, i in model.p_vec2:
        p_vec2[j][i] = pyo.value(model.p_vec2[j, i])

    e_vec2 = np.zeros((m,n+1))
    for j, i in model.e_vec2:
        e_vec2[j][i] = pyo.value(model.e_vec2[j, i])

    # Assert if the losses match the storage system model
    for e_vec, p_vec in zip( [e_vec1, e_vec2], [p_vec1, p_vec2]):
        # Calculate the losses from the solution of the optimization problem
        losses = np.zeros((m,n))
        for j in range(m):
            for i in range(n):
                losses[j][i] = -(e_vec[j][i+1] - e_vec[j][i] + dt*p_vec[j][i])/dt
        # Check that the losses match their expected values
        tol = 1e-4 # same tolerance as for the optimization algorithm
        verifiedModel = True
        for j in range(m):
            for los, pow in zip(losses, p_vec):
                    wdw_in = np.where(pow < 0)
                    wdw_out = np.where(pow >=0)

                    diff_in = los[wdw_in] + (1-eta1_in)* pow[wdw_in]
                    diff_out = los[wdw_out] - (1-eta1_out)/eta1_out* \
                                                pow[wdw_out]

                    error_in = sum( eps**2 for eps in diff_in)
                    error_out = sum( eps**2 for eps in diff_out)
                    if error_in > tol or error_out > tol:
                        verifiedModel = False
        if not verifiedModel:
            print('Error above tolerance in solve_dispatch_pyomo:', max(error_in, error_out))

    
    return p_vec1, e_vec1,  p_vec2, e_vec2, bin, results.solver.status