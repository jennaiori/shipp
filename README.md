# SHIPP: Sizing optimization for HybrId Power Plants

## Description
SHIPP is used for studying and designing hybrid power plants, i.e. power plants combining one or more renewable energy production with energy storage systems.

SHIPP is in development. Its capabilities are currently limited to sizing and operation of storage systems only. 


## Installation
*Installation instructions will be written once the package is published*

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

The optimization problem is formulated and solved using a command of the form:
```python
os = solve_lp_pyomo(price, production1, production1, storage1, storage2, discount_rate, n_year, p_min, p_max, n)
```
where the discount rate and the number of years `n_year` are used for the calculation of the NPV. The parameters `p_min` and `p_max` are used to describe the constraints for the maximum and minimum power production. The number of time steps is `n`.

This results in an `OpSchedule` object describing the operation schedule of the power plant, with the following members:
- `production_list`: a list of `Production` objects corresponding to the input objects of the optimization problem
- `storage_list`: a list of `Storage` objects corresponding to the input objects of the optimization problem. However, their power and energy capacity correspond to the optimal design. In addition, we use a round-trip efficiency accounted during discharge, instead of separate charging and discharging efficiency.
- `production_p`: a list of the power production for the objects in `production_list`. In case of curtailment, the first production unit assumes all the curtailment and the second production unit operates at full power.
- `storage_p`: list of optimal power operation (charging/discharging) for the storage objects 
- `storage_e`: list of optimal energy level evolution for the storage objects  
- `losses`: list of power losses corresponding to the storage objects.
- `power_out`: total power to the grid for the production and storage units
- `npv`: NPV for the total system

The operation schedule can be visualized using the following commands:
- `os.plot_powerflow()`: line plot of the power production and the energy level evolution
- `os.plot_powerout(xlim = [start_time, end_time])`: bar plot of the power send to the grid

An example case is given in `examples/example1.ipynb`.

## Future developments
- Remove dependency on mosek
- Expand optimization problem definition to an arbitrary number of production and storage objects.
- Test implementation with gurobi and cplex.
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

The file `environment.yml` can be used to create the conda environment to run the code.


## Authors and acknowledgment
This project is developed by Jenna Iori at Delft University of Technology and is part of the Hollandse Kust Noord wind farm innovation program. Funding was provided by CrossWind C.V.

## License

The code is currently closed source.