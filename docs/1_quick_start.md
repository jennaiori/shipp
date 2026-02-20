# Quick start

This page describes how to get started with the SHIPP package 🚢. 


## Installation

SHIPP is available as a Python package on PyPI and can be installed using pip.

```shell
pip install shipp
```

## Set-up

The methods implemented in the package require timeseries of power and price as inputs. Such timeseries can be generated artificially or via the APIs of [renewables.ninja](https://www.renewables.ninja) and ENTSO-E. 

```{warning}
   Please ensure that the price timeseries does not contain negative values. Due to the custom LP formulation of the dispatch optimization problem, negative prices are not supported.  
```

```python
# Artifical generation of timeseries
n = 1000 # Number of time steps
frequency_power = 0.5 
frequency_price = 5
mean_power = 50
price_low = 40
price_high = 60
p_max = 100

time = np.arange(0, n)/24

# The power is represented by a sine function of time
data_power = mean_power*(1.0+np.sin(time * 2*np.pi * frequency_power))

# The price is represented by the combinaison of a sine function of time and a random variation
data_price = np.random.uniform(price_low, price_high, n) + \
        10*np.sin(np.arange(0, n)/24 * 2*np.pi * frequency_price)

# Retrieval of power and price timeseries via renewables.ninja and entso-e
date_start = '2019-01-01'
date_end = '2019-02-02'
latitude = 52.52
longitude = 13.405
p_max = 100

# API tokens
token_rninja = 'replace_by_your_own'
token_entsoe = 'replace_by_your_own'

data_power, data_price = get_power_price_data(token_rninja, token_entsoe, date_start, date_end, latitude, longitude, capacity = 100*1e3)
```
The raw data is then stored in a {py:obj}`TimeSeries <timeseries.TimeSeries>` object, with the information of the time step `dt`. 

```python
dt = 1 # time step in hour
power_ts = TimeSeries(data_power, dt)
price_ts = TimeSeries(data_price, dt)
```

A hybrid power plant is described with a combinaison of power generation technologies, described by the {py:obj}`Production <components.Production>` class, and storage technologies, described by {py:obj}`Storage <components.Storage>` class. 

A {py:obj}`Production <components.Production>` object describes a renewable energy production (e.g. wind or solar PV) through
- its power production in time `power`, stored in a `TimeSeries` object.
- its cost per power capacity `p_cost`.

A {py:obj}`Storage <components.Storage>` object describes an abstract storage systems through: 
- its cost per energy and power capacity `e_cost` and `p_cost`
- its energy and power capacity `e_cap` and `p_cap`
- its efficiency in charge and discharge `eff_in` and `eff_out`. Note that only constant efficiencies are considered here.
- its depth of discharge `dod`.

Below is an example of initialization.
```python
# Production initialization
p_cost_res = 3000  # cost per installed capacity of the renewable energy source [Eur/MW]
prod = Production(power_ts, p_cost = p_cost_res)

# Storage initialization
p_cost = 150*1e3  # cost per power capacity [Eur/MW]
e_cost = 75 * 1e3 # cost per energy capacity [Eur/MWh]
eta = 0.85 #Round trip efficiency, here applied in discharge.
dod = 0.9 # Depth of discharge
e_cap = 20 # Energy capacity [MWh]
p_cap = 10 # Power capacity [MW]
stor = Storage(e_cap = e_cap, p_cap = p_cap, eff_in = 1, eff_out= eta, e_cost = e_cost, p_cost = p_cost)
```

## Simple dispatch problem

The optimal dispatch strategy for revenue maximization can be calculated by solving the dispatch optimization problem implemented in {py:func}`solve_lp_sparse <kernel.solve_lp_sparse>` or {py:func}`solve_lp_pyomo <kernel_pyomo.solve_lp_pyomo>`. Here, the power and price are assumed to be known perfectly for the entire time window.

The method requires the following additional inputs:
```python
n = min(len(data_price), len(data_power)) # Length of the simulation
p_min = 0
p_max = 100 # maximum allowed delivered power, grid connection capacity [MW]
discount_rate = 0.03
n_year = 20
prod_null = Production(TimeSeries([0 for _ in range(n)], dt), 0)  # Creation of a null object
stor_null = Storage(e_cap = 0, p_cap = 0, eff_in =1, eff_out=1, e_cost = 0, p_cost = 0)  # Creation of a null object
```
```{note}
The discount rate and number of years are required because the underlying objective function is the net present value (NPV). However, in this case, the capacity of the storage system (and therefore the corresponding costs) are fixed. Thus, maximizing the NPV is equivalent to maximizing revenues.
```

The dispatch optimization problem can be solved with the built-in solver in scipy, `scipy.optimize.linprog` with the following command:

```python
os = solve_lp_sparse(price_ts, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, fixed_cap = True)
```

However, this solver performs poorly with large problems. Instead, it is recommended to use off-the-shelf solvers throught the pyomo interface with the following command, where the parameter `name_solver` refers to a solver compatible with pyomo, for example 'mosek', 'cplex', 'gurobi'.

```python
os = solve_lp_pyomo(price_ts, prod, prod_null, stor, stor_null, discount_rate, n_year, p_min, p_max, n, name_solver, fixed_cap = True)
```


The optimization results are stored in a {py:obj}`OpSchedule <components.OpSchedule>` object describing the operation schedule of the power plant. You can display the results with the following commands:

```python
print('Storage revenues\t{} Eur'.format(os.revenue_storage))
print('Total revenues\t{} Eur'.format(os.revenue))

# Generate a figure with the storage power and state of charge
time_vec = np.arange(n)*1/24
fig, ax = plt.subplots(2,1)
ax[0].plot(time_vec,os.storage_p[0].data)
ax[0].set_xlabel('Time [days]')
ax[0].set_ylabel('Storage Power [MW]')

ax[1].plot(time_vec, os.storage_e[0].data)
ax[1].set_xlabel('Time [days]')
ax[1].set_ylabel('Storage state of charge [MWh]')
```
Additional details can be obtained with the other members of the class {py:obj}`OpSchedule <components.OpSchedule>`


## Online dispatch operation

Instead of using perfect power foresight, SHIPP implements a method to simulate the storage operation using a rolling-window, i.e. the storage dispatch is calculated every time step with updated forecast information.

The forecast data must be stored in a 3-dimensional list format, where the first index refers to the time step, the second index refers to the trajectory number (in case of ensemble forecast) and the third index is the lead-time of the forecast. Thus, from the power observation `data_power`, a forecast signal can be built as follows

```python
n_for = 12 # Maximum forecast lead time in number of time steps
n = 48 # Number of time steps in the simulation
forecast = [ [[p for p in data_power[init_index:init_index+n_for]]]  for init_index in range(0, n)] 
```

Using the forecast as an input, the method {py:func}`run_storage_operation <kernel_pyomo.run_storage_operation>` simulates the storage operation for a given number of time steps

```python
dt = 1 # Time step in hour
e_start = stor.e_cap # Initial state of charge
p_min = 0
p_max = 100


res = run_storage_operation('forecast', data_power, data_price, p_min, p_max, stor, e_start, n_for, n, dt, forecast = forecast, name_solver = name_solver)
```

Here, the function returns a dictionary containing the storage power and energy time series and the revenues. Results can be displayed with the following commands:

```python
print('Storage revenues\t{} Eur'.format(res['revenues']))

# Generate a figure with the storage power and state of charge
time_vec = np.arange(n)*1/24
fig, ax = plt.subplots(1,2)
ax[0].plot(time_vec, res['power'])
ax[0].set_xlabel('Time [days]')
ax[0].set_ylabel('Storage Power [MW]')

ax[1].plot(time_vec, res['energy'][:-1])
ax[1].set_xlabel('Time [days]')
ax[1].set_ylabel('Storage state of charge [MWh]')
```

## Baseload power production

A minimum baseload power production can be enforced in the dispatch optimization problem through the parameter `p_min`. 

In the methods {py:func}`solve_lp_sparse <kernel.solve_lp_sparse>` and {py:func}`solve_lp_pyomo <kernel_pyomo.solve_lp_pyomo>`, the problem can become infeasible if there is no sufficient storage to satisfy the baseload power production with a 100% reliability. 

In the method {py:func}`run_storage_operation <kernel_pyomo.run_storage_operation>`, the optimization problem includes reliability of the baseload production in the objective function. This is done through a penalty term, tuned by the input parameter `mu`. The reliability of the baseload power production in the results is stored in `res['reliability']`.

## Ramp-limitation

Similarly, a ramp-limit can be enforced in the dispatch optimization problem, here with the parameter `dp_lim`. 

```{note}
   The ramp-limitation contraint is not implemented in ``solve_lp_sparse``.
```