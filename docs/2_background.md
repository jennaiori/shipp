# Mathematical background


## Hybrid power plant modelling

A hybrid power plant can be defined as the "combination of two or more electricity generation and/or storage technologies, providing coordinated electrical power services that are connected at a single point." ([IEA Task 50 - WP1](https://iea-wind.org/task50/t50-wp1/)). 

### Components and characteristics

In SHIPP, the components of the hybrid power plant are divided in power generation components (wind farm, solar farm, etc.) and storage systems components (battery, etc.). 

Each power generation component $g$ is characterized by:
- $\boldsymbol{p}^g$: the time series of power generation during the considered time window
- $\bar{P}^g$: its rated power
- $\lambda_p^g$: its cost per MW of rated power

Each storage compoment $s$ is characterised by:
- $\bar{P}^s$: its power capacity
- $\bar{E}^s$: its energy capacity
- $\eta_\text{in}^s$, $\eta_\text{out}^s$: its efficiency in charge and discharge
- $d^s$: its depth of discharge
- $\lambda_p^s$: its cost per MW of power capacity
- $\lambda_e^s$: its cost per MWh of energy capacity
- $\boldsymbol{p}^s$: the time series of its power (in charge and discharge)
- $\boldsymbol{e}^s$: the time series of its state of charge

The hybrid power plant is characterized by:
- the number and type of each component
- $\bar{P}$: its rated power (corresponding to the capacity of the point of connection)
-  $\boldsymbol{p}$: the time series of the power sent to the grid 
-  $\boldsymbol{p}^c$: the time series of curtailed power


### Power delivery model

SHIPP models the power delivered to the grid as the sum of the power produced by the generation components, the power in charge/discharge of the storage components, minus the curtailed power,

$$ \boldsymbol{p} = \sum_{g} \boldsymbol{p}^g +  \sum_{s} \boldsymbol{p}^s - \boldsymbol{p}^c  $$

We assume that the time increment $\Delta t$ is sufficiently large to neglect transient effects.

(storage_system_model)=
### Storage system model

The operation of the storage systemm is described with the power $\boldsymbol{p}^s$ and energy $\boldsymbol{e}^s$ time series. The following convention is used: the power is negative during charge (*in*) and positive during discharge (*out*).
The power and energy time series are bounded by the power and energy capacities of the component:

$$ \forall i \in[0,n[, \quad  -\bar{P}^s \leq p_i^s \leq \bar{P}^s$$
$$ \forall i \in[0,n], \quad  \bar{E}^s (1 - d^s) \leq e_i^s \leq \bar{E}^s$$


Considering a time window of $n$ time steps with a discretization $\Delta t$, the power and energy time series satisfy the charge/discharge model of the storage for $i \in [0, n[$


```{math} 
e^s_{i+1} - e^s_{i} = \begin{cases}
  - \Delta t \ \eta^s_\text{in} \  p^s_i & \text{if } p^s_i \leq 0 \\
  - \Delta t \dfrac{1}{\eta^s_\text{out}}  p^s_i & \text{else}.
\end{cases}
```

Considering realistic storage system efficiency ($\eta_\text{in}^s \eta_\text{out}^s <1$), energy losses occur during cycles. The total can be calculated as

```{math}
L^s = -\Delta t \sum_{i=0}^{n-1} e^s_{i+1} - e^s_{i} + \Delta t p^s_i
```

```{note}
Degradation is not currently implemented in the code.
```

## Performance metrics

The design and operation of the hybrid power plant can be assessed using different performance metrics, covering revenues, costs and reliability.

### Energy and losses

The hybrid power plant is characterised by its annual energy production (AEP) and grid utilization factor (GUF)

```{math}
 \text{AEP} = \dfrac{8760 \Delta t}{n} \mathbb{1}^T \cdot \boldsymbol{p} \Delta t 
 ```

```{math}
 \text{GUF} = \dfrac{1}{n} \dfrac{\mathbb{1}^T \cdot \boldsymbol{p}}{\bar{P}} 
 ```

The energy losses in the power plant come from curtailment and from the inefficiency of the charge/discharge cycle in the storage system.

```{math}
 L_\text{tot} = \mathbb{1}^T \cdot (\sum_{g} \boldsymbol{p}^g - \boldsymbol{p}) \Delta t  = L^c + \sum_{s} L^s
 ```
with $L^c = \mathbb{1}^T \cdot \boldsymbol{p}^c \Delta t$.
 

### Revenues

The revenues of the hybrid power plant are calculated as the dot product between a time series of electricity price $\boldsymbol{\lambda}$ and the time series of power delivered to the grid, 

```{math}
 c_\text{R} = \boldsymbol{\lambda}^T \cdot \boldsymbol{p}
 ```

This simple revenue model does not account for the different bidding stages in the electricity markets. It only captures the variability of electricity price in time and its correlation with renewable energy production. The time series of day-ahead market price can be used for $\boldsymbol{\lambda}$, as first-order estimation of how the value of electricity varies in time.

For studying the performance of the storage system specifically, it is relevant to calculate the difference between revenues with and without storage,


```{math}
 \Delta \text{R} = \boldsymbol{\lambda}^T \cdot ( \text{min}(\bar{P}, \sum_{g} \boldsymbol{p}^g) - \boldsymbol{p})
 ```
When the grid connection capacity is above or equal the installed power capacity (i.e., no overplanting), the added revenues can be simplified as

```{math}
\Delta \text{R} = \boldsymbol{\lambda}^T \cdot (\sum_{s} \boldsymbol{p}^s - \boldsymbol{p}^c)
 ```

Finally, arbitrage revenues generated by the storage system components are calculated as

```{math}
\text{R}_a = \boldsymbol{\lambda}^T \cdot (\sum_{s} \boldsymbol{p}^s)
 ```

### Net Present Value

Profitability of the hybrid power plant can be estimated using the net present value (NPV) and internal rate of return (IRR). These two metrics take into account the capital expenditures of each component, the project lifetime and discounts revenues. The NPV can be estimated with

```{math}
\text{NPV} = -\sum_{g} \lambda^g_P \bar{P}^g - \sum_{s} (\lambda^s_P \bar{P}^s + \lambda_E^s \bar{E}^s) + \sum_{k=1}^m \dfrac{\text{R}}{(1+r)^k}
 ```

where $m$ is the number of years of operation of the project and $r$ the discount rate. Here we assume that the revenues $R$ estimated for one year are representative of the entire lifetime of the project, and that all capital expenditures happen at year 0.

The internal rate of return is the discount rate for which $\text{NPV} = 0$.

In the presence of storage system components, it is relevant to calculate the *added NPV*, i.e., the contribution of storage system to the total NPV. It is calculated as

```{math}
\Delta \text{NPV} = -\sum_{s} (\lambda^s_P \bar{P}^s + \lambda_E^s \bar{E}^s) + \sum_{k=1}^m \dfrac{\Delta \text{R}}{(1+r)^k}
 ```

It is possible to use a more precise formula of the NPV, taking into account the operational expenditures and the timing of the storage system replacement.


### Reliabity

In SHIPP, the term reliability refers to the percentage of time where a specific dispatch constraint  is satified (see below). A dispatch constraint can be expressed as a non-linear equation of the delivered power at each time step,  $\boldsymbol{g}(p_i) \leq 0$. Then, the associated reliability $r$ is expressed as

```{math}
r = \dfrac{1}{n}\sum_{i=0}^{n-1} z_i, \text{   where  } z_i = \begin{cases}
  1 & \text{if } \boldsymbol{g}(p_i) \leq 0, \\
  0 & \text{else}.
\end{cases}

```

## Dispatch constraints


The power delivered should respect bounds dictated by the grid connection (assuming the storage systems cannot charge from the grid):

$$ \forall i, \ 0 \leq\sum_{g} p^g_i + \sum_{s} p^s_i - p^c_i \leq \bar{P}  $$

SHIPP allows additional dispatch constraints to be enforced on the power delivery:
- A minimum baseload power $P_\text{bl}$
- A maximum power ramp $\delta P_\text{rl}$

The baseload constraint is expressed as a lower bound on the delivered power, 

$$ \forall i, \ P_\text{bl} \leq\sum_{g} p^g_i + \sum_{s} p^s_i - p^c_i. $$

In the presence of a ramp limit, the constraint is applied to the difference of power between two time steps

$$ \forall i, \ - \delta P_\text{rl} \leq\sum_{g} (p^g_{i+1} - p^g_i ) + \sum_{s} (p^s_{i+1} - p^s_i) - (p^c_{i+1} - p^c_i)  \leq \delta P_\text{rl} $$