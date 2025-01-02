# Problem Description

## Hybrid power plant components

A hybrid power plant can be defined as the "combination of two or more electricity generation and/or storage technologies, providing coordinated electrical power services that are connected at a single point." (IEA Task 50 - WP1). In SHIPP, the components of the hybrid power plant are divided in power generation components (wind farm, solar farm, etc.) and storage systems components (battery, etc.). 

Each power generation component $g$ is characterized by:
- $\boldsymbol{p}^g$: the time series of power generation during the considered time window
- $\bar{P}^g$: its rated power
- $\lambda_p^g$: its cost per MW of rated power

Each storage compoment $s$ is characterised by:
- $\bar{P}^s$: its power capacity
- $\bar{E}^s$: its energy capacity
- $\eta_\text{in}^s$, $\eta_\text{out}^s$: its efficiency in charge and discharge
- $\lambda_p^s$: its cost per MW of power capacity
- $\lambda_e^s$: its cost per MWh of energy capacity
- $\boldsymbol{p}^s$: the time series of its power (in charge and discharge)
- $\boldsymbol{e}^s$: the time series of its state of charge

The hybrid power plant is characterized by:
- the number and type of each component
- $\bar{P}$: its rated power (corresponding to the capacity of the point of connection)
-  $\boldsymbol{p}$: the time series of the power sent to the grid 
- (optional) $P_\text{bl}$: its required baseload level

## Storage system model

The cost $c^s$ of each storage component is calculated directly from its energy and power capacity,
$c^s = \lambda_P^s \bar{P}^s  + \lambda_E^s \bar{E}^s$.

The operation of the storage systemm is described with the power $\boldsymbol{p}^s$ and energy $\boldsymbol{e}^s$ time series. The following convention is used: the power $p$ is negative during charge (*in*) and positive during discharge (*out*).


Considering a time window of $n$ time steps with a discretization $\Delta t$, the power and energy time series satisfy the charge/discharge model of the storage for $i \in [0, n[$

$$ e^s_{i+1} - e^s_{i} =  - \Delta t \ \eta^s_\text{in} \  p^s_i \quad \text{if } p^s_i \leq 0, $$
$$ e^s_{i+1} - e^s_{i} =  - \Delta t \dfrac{1}{\eta^s_\text{out}}  p^s_i \quad \text{else}. $$

Furtermore, the power and energy time series are bounded by the power and energy capacities of the component:

$$ \forall i \in[0,n[, \quad  -\bar{P}^s \leq p_i^s \leq \bar{P}^s$$
$$ \forall i \in[0,n], \quad  0 \leq e_i^s \leq \bar{E}^s$$

Here, we assume that the minimum allowed state of charge is zero. Furthermore, degradation of the storage in time is not modeled.


## Design problem

We want to size the storage system, in terms of its power and energy capacity, with the objective to maximize the net present value (NPV) of the system. 

(Optional) This design problem can be solved with a baseload constraint, i.e. the power plant is required to produce a minimum power.


The objective function of the (minimization) problem is:

```{math}
 c(\boldsymbol{x}) = \sum_{s} (\lambda^s_P \bar{P}^s + \lambda_E^s \bar{E}^s) - \sum_{k=1}^m \sum_{s}\dfrac{\boldsymbol{\lambda}_\text{DAM}^T \cdot p^s}{(1+r)^k}
 ```

where $m$ is the lifetime of the project, $r$ the discount rate and $\lambda_\text{DAM}$ is the time series of electricity price on the day-ahead market.

