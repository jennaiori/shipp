# SHIPP: Sizing optimization for HybrId Power Plants

## Description
SHIPP is used for studying and designing hybrid power plants, i.e. power plants combining one or more renewable energy production with energy storage systems.

SHIPP is in development. Its capabilities are currently limited to sizing and operation of storage systems only. 

For any question about the code, please contact Jenna Iori at j.iori@tudelft.nl
## Installation
The package can be installed locally using pip after cloning the repository.

```python
pip install path-to-shipp-folder
```


## Usage

SHIPP relies on two classes to describe a sizing optimization problem:
- a `Production` object describe a renewable energy production (e.g. wind or solar PV) through
    - its power production in time `power`. Note that in the optimization problem, the power can be curtailled.
    - its cost per power capacity `p_cost`
- a `Storage` object describe abstract storage systems through: 
    - its cost per energy and power capacity `e_cost` and `p_cost`
    - its energy and power capacity `e_cap` and `p_cap`
    - its efficiency in charge and discharge `eff_in` and `eff_out`. Note that only constant efficiencies are considered here.

Finally, the optimization problem requires the time series of electricity prices on the day-ahead market.

### Solving the optimization problem
The optimization problem is formulated and solved using a command of the form:
```python
os = solve_lp_pyomo(price, production1, production1, storage1, storage2, discount_rate, n_year, p_min, p_max, n, name_solver)
```
where the discount rate and the number of years `n_year` are used for the calculation of the NPV. The parameters `p_min` and `p_max` are used to describe the constraints for the maximum and minimum power production. The number of time steps is `n`. The parameter `name_solver` refers to a solver compatible with pyomo, for example 'mosek', 'cplex', 'gurobi'.

The problem can be solved with the built-in solver in scipy, `scipy.optimize.linprog` with the following command:

```python
os = solve_lp_sparse_sf(price, production1, production1, storage1, storage2, discount_rate, n_year, p_min, p_max, n)
```
However, this solver uses a dense matrix representation and is not appropriate for large problems (`n`> 3600).

### Accessing the results of the optimization
This results in an `OpSchedule` object describing the operation schedule of the power plant, with the following members:
- `production_list`: a list of `Production` objects corresponding to the input objects of the optimization problem
- `storage_list`: a list of `Storage` objects corresponding to the input objects of the optimization problem. However, their power and energy capacity correspond to the optimal design.
- `production_p`: a list of the power production for the objects in `production_list`. In case of curtailment, the first production unit assumes all the curtailment and the second production unit operates at full power.
- `storage_p`: list of optimal power operation (charge/discharge) for the storage objects 
- `storage_e`: list of optimal energy level evolution for the storage objects  
- `losses`: list of power losses corresponding to the storage objects.
- `power_out`: total power to the grid for the production and storage units
- `npv`: NPV for the total system

The operation schedule can be visualized using the following commands:
- `os.plot_powerflow()`: line plot of the power production and the energy level evolution
- `os.plot_powerout(xlim = [start_time, end_time])`: bar plot of the power send to the grid


An example case is given in `examples/example1.py`.

## Future developments
- Publish package on PyPI
- Expand optimization problem definition to an arbitrary number of production and storage objects.
- Include the lifetime of storage systems in the `Storage` objects.
- Remove dependency on class `TimeSeries`

## Dependencies
The code relies on the following python packages:
- numpy
- numpy-financial
- pandas
- scipy
- matplotlib
- requests
- ipykernel
- pyomo 
- entsoe-py

Furthermore, a valid access or license to a solver compatible with pyomo (MOSEK, CPLEX, Gurobi, etc.) is recommended to solve large problems (see more information here: https://www.pyomo.org/).

## Authors and acknowledgment
This project is developed by Jenna Iori at Delft University of Technology and is part of the Hollandse Kust Noord wind farm innovation program. Funding was provided by CrossWind C.V.

The code is release under the Apache 2.0 License (see License.md).

## Copyright notice: 

Technische Universiteit Delft hereby disclaims all copyright interest in the program “SHIPP” (a design optimization software for hybrid power plants) written by the Author(s). 

Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2024, Jenna Iori