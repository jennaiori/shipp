# Implementation

SHIPP currently implements the analysis of two storage system components and two power generation components. 
The code has so far only be used for an hourly time step.

## Formulation of the storage system model

The storage system model linking power to stored energy is a piece-wise linear function (see [](storage_system_model)) and cannot be implemented directly in a linear or mixed-integer linear optimization problem. In SHIPP, three alternative formulations of the storage system model constraints are implemented:
- a mixed-integer linear (`milp`) formulation where integer variables are used to indicate if the storage is charging or discharging. Since the equations are enforced exactly, any feasible point of the optimization validates the model.
- a linear formulation using a relaxed form of the MILP (`lp-alt`). In this case, the storage system is only valid at the optimum. Feasible points for the optimization problem do not necessarily validate the storage model.
- a linear formulation (`lp`) where the storage power is divded into a charging and discharging terms. Here as well, the validity of the storage model is not guaranteed due to the possibility of simultaneous charge and discharge.


### MILP
The storage system model is enforced using a binary variable at each time step $i$, denoted by $z_i$. The storage power is divided into two components: $p^{s+}$ in dischage and $p^{s-}$. The constraints are implemented using a big-M form,

$$ e^s_{i+1} - e^s_{i} =  \Delta t \ \eta^s_\text{in} \  p^{s-}_i - \Delta t \dfrac{1}{\eta^s_\text{out}}  p^{s+}_i  $$
$$ 0 \leq p^{s-}_i \leq M z^s_i $$
$$ 0 \leq p^{s+}_i \leq M (1 - z^s_i) $$

We use the convention $z^s_i=0$ for charge and $z^s_i=1$ for discharge. In addition, the following bounds are imposed on the storage power, 
$$ 0 \leq p^{s-}_i \leq \bar{P}^s $$
$$ 0 \leq p^{s+}_i \leq \bar{P}^s $$

<!-- It is also possible to do this implementation without separating the power into two components, with

$$ -M z^s_i \leq  e^s_{i+1} - e^s_{i} + \Delta t \ \eta^s_\text{in} \  p^s_i \leq M z^s_i,$$
$$ -M(1-z^s_i) \leq e^s_{i+1} - e^s_{i} + \Delta t \dfrac{1}{\eta^s_\text{out}}  p^s_i  \leq M(1-z^s_i), $$
$$ -M(1-z^s_i)  \leq p^s_i \leq M z^s_i. $$ -->



### LP-alt
In the LP-alt formulation, the storage system model is represented by two inequality constraints,

$$ e^s_{i+1} - e^s_{i} \leq  - \Delta t \ \eta^s_\text{in} \  p^s_i, $$
$$ e^s_{i+1} - e^s_{i} \leq - \Delta t \dfrac{1}{\eta^s_\text{out}}  p^s_i. $$

For this implementation to be representative, one of the inequality constraint must always be active at the optimum. As a result, the objective function must maximize the power at all time steps ($p^s_i$). This means that negative prices should not be used.

The bounds on the storage power are set as $ -\bar{P}^s \leq p^{s-}_i \leq \bar{P}^s$.
### LP

The LP formulation is very similar to the MILP one. It consists of the same bounds on the power time series and the same equality constraint, 

$$ e^s_{i+1} - e^s_{i} =  \Delta t \ \eta^s_\text{in} \  p^{s-}_i - \Delta t \dfrac{1}{\eta^s_\text{out}}  p^{s+}_i  $$

Without integer variables, it is possible to have simultaneous charge and discharge at the optimum, $p^{s-}_i p^{s+}_i != 0$. In this case, there is a risk that the storage model is not respected. A regularization term in the objective function should be used as a mitigation measure.

## Perfect foresight optimization

The functions `solve_lp_sparse` and `solve_lp_pyomo` implement a dispatch optimization where information about power and price is known perfectly for the entire simulation horizon. They implement an integrated design problem, where design and operation are found simulatenously. It is also possible to solve the problem for dispatch only by setting the parameter `fixed_cap` to `True`.

The design variables of the problem $\boldsymbol{x}$ are the storage system components energy and power capacities, the time series of storage power and energy, and the time series of curtailed power. If `fixed_cap = True`, the storage capacities are either removed from the list of design variables or fixed to their initial values. Depending on the problem formulation, the storage system power is represented by one single variable ($\boldsymbol{p}^s$) or by its components in charge and discharge.


#### In `solve_lp_sparse`

In `solve_lp_sparse`, the implemented optimization problem is  

```{math}
\begin{align}
\text{min} &&&  f(\boldsymbol{x}) \\
\text{s.t.} &&&  P_\text{bl} \leq  p_i \leq \bar{P} &&& i = 0,..., n-1 \\
  &&& e^s_0 = e^s_n, &&& s \in \{1,2\} \\
  &&& (1-d)\bar{E} \leq e^{s}_i \leq \bar{E}^s &&& i = 0,..., n \\
  &&& 0 \leq p^c_i \leq \sum_g p_i^g &&& i = 0,..., n-1 \\
  &&&  \text{Storage model} \\
  &&& \text{Storage capacity bounds} 
\end{align}
```
where the storage model constraints depend on the formulation, as described above. The storage capacity bounds depend on the parameter `fixed_cap`. If `fixed_cap = False`, one has


```{math}
\begin{align}
0 \leq \bar{P}^s \leq \bar{P}^s_\text{max}, &&& s \in \{1,2\} \\
0 \leq \bar{E}^s \leq \bar{E}^s_\text{max}, &&& s \in \{1,2\}
\end{align}
```

where $ \bar{P}^s_\text{max}$ and $\bar{E}^s_\text{max}$ are either equal to the initial capacities of the storage system ( $\bar{P}^s_\text{init}, \bar{E}^s_\text{init}$) or to an arbitrary large value if a capacity is set to `None`.

If `fixed_cap = True`, one has instead
```{math}
\begin{align}
 \bar{P}^s = \bar{P}^s_\text{init}, &&& s \in \{1,2\} \\
\bar{E}^s = \bar{E}^s_\text{init}, &&& s \in \{1,2\} 
\end{align}
```
The problem can be represented in standard form, as 

```{math}

\begin{align}
\text{min} &&&  \boldsymbol{c}_\text{obj}^T \boldsymbol{x} \\
\text{s.t.} &&&  \mathbf{A}_\text{eq} \boldsymbol{x} = \boldsymbol{b}_\text{eq} \\
            &&&  \mathbf{A}_\text{ineq} \boldsymbol{x} \leq \boldsymbol{b}_\text{ineq} \\
            &&&   \boldsymbol{x}_\text{lb} \leq \boldsymbol{x} \leq \boldsymbol{x}_\text{ub} 
\end{align}
```

The terms $\boldsymbol{c}_\text{obj}, \mathbf{A}_\text{eq}, \boldsymbol{b}_\text{eq}, \mathbf{A}_\text{ineq}, \boldsymbol{b}_{ineq}, \boldsymbol{x}_\text{lb}, \boldsymbol{x}_\text{ub}$ are constructed with the routines `build_lp_obj_npv`, `build_lp_obj_revenues` and `build_lp_cst_sparse` and used as input in the solver `scipy.linprog`. 

```{note}
To reduce memory use and due to the large size of the problem, all matrices are stored in a sparse format.
```

The objective function aims to either maximize the added NPV or added revenues from the storage components. It includes three regularization parameters: $\alpha$, $\beta$ and $\epsilon$. The parameters $\alpha$ and $\beta$ are used to regularize the curtailed power with respect to the storage power. The parameter $\epsilon$ is only relevant for the `lp` formulation and helps avoiding simulatenous charging and discharging in the storage systems.

For the `lp` and `milp` formulation, the storage power is divided into charge $\boldsymbol{p}^{s-}$ and discharge $\boldsymbol{p}^{s+}$. The objective function of the problem is:

- For the integrated design problem: 
```{math}
  f_\text{NPV}(\boldsymbol{x}) = \sum_{s} (\lambda^s_P \bar{P}^s + \lambda_E^s \bar{E}^s) - \dfrac{8760}{n}\sum_{k=1}^m \dfrac{\boldsymbol{\lambda}^T \cdot (\sum_{s} (\boldsymbol{p}^{s+} - \boldsymbol{p}^{s-}) - \alpha \boldsymbol{p}^c)}{(1+r)^k}  + \beta \mathbb{1}^T\cdot \boldsymbol{p}^c + \sum_{s} \epsilon \mathbb{1}^T\cdot ( \boldsymbol{p}^{s+} +  \boldsymbol{p}^{s-})
```
- For the dispatch only problem:
```{math}
  f_R(\boldsymbol{x}) = - \boldsymbol{\lambda}^T \cdot (\sum_{s} (\boldsymbol{p}^{s+} - \boldsymbol{p}^{s-}) - \alpha \boldsymbol{p}^c) + \beta \mathbb{1}^T\cdot \boldsymbol{p}^c + \sum_{s} \epsilon \mathbb{1}^T\cdot ( \boldsymbol{p}^{s+} +  \boldsymbol{p}^{s-})
```


For the `lp-alt` formulation, the objective function is:
- For the integrated design problem: 
```{math}
  f_{\text{NPV}, \text{alt}}(\boldsymbol{x}) = \sum_{s} (\lambda^s_P \bar{P}^s + \lambda_E^s \bar{E}^s) - \dfrac{8760}{n}\sum_{k=1}^m {\boldsymbol{\lambda}^T \cdot (\sum_{s} \boldsymbol{p}^{s} - \alpha \boldsymbol{p}^c)}{(1+r)^k} + \beta \mathbb{1}^T\cdot \boldsymbol{p}^c
```
- For the dispatch only problem:
```{math}
  f_{R, \text{alt}}(\boldsymbol{x}) = - \boldsymbol{\lambda}^T \cdot (\sum_{s} \boldsymbol{p}^{s} - \alpha \boldsymbol{p}^c) + \beta \mathbb{1}^T\cdot \boldsymbol{p}^c
```

The vector of design variable changes depending on the problem formulation and the parameter `fixed_cap`, as reported in the table below.

| Formulation | Objective function | `fixed_cap`| Design variables $\boldsymbol{x}$ |
| ----------- | ------------------ |----------- | ---------------- |
|  `milp`     | $f_\text{NPV}$      | False  |$\boldsymbol{p}^c, (\bar{P}^s, \bar{E}^s, \boldsymbol{p}^{s+}, \boldsymbol{p}^{s-}, \boldsymbol{e}^s, \boldsymbol{z}^s_i)_{s\in{1,2}}$  |
| `milp`     | $f_R$        | True  |$\boldsymbol{p}^c, (\boldsymbol{p}^{s+}, \boldsymbol{p}^{s-}, \boldsymbol{e}^s, \boldsymbol{z}^s_i)_{s\in{1,2}}$ |
|  `lp-alt`     | $f_{\text{NPV}, \text{alt}}$      | False  |$\boldsymbol{p}^c, (\bar{P}^s, \bar{E}^s, \boldsymbol{p}^s, \boldsymbol{e}^s)_{s\in{1,2}}$  |
| `lp-alt`     | $f_{R, \text{alt}}$        | True  |$\boldsymbol{p}^c, (\boldsymbol{p}^s, \boldsymbol{e}^s)_{s\in{1,2}}$ |
|  `lp`     | $f_\text{NPV}$      | False  |$\boldsymbol{p}^c, (\bar{P}^s, \bar{E}^s, \boldsymbol{p}^{s+}, \boldsymbol{p}^{s-}, \boldsymbol{e}^s)_{s\in{1,2}}$  |
| `lp`     | $f_R$        | True  |$\boldsymbol{p}^c, (\boldsymbol{p}^{s+}, \boldsymbol{p}^{s-}, \boldsymbol{e}^s)_{s\in{1,2}}$ |


### In `solve_lp_pyomo`
The dispatch optimization in `solve_lp_pyomo` is currently only implemented using the `lp-alt` formulation. The design variables of the problem are the storage system components energy and power capacities, the time series of storage power and energy, and the time series of curtailed power, i.e., $\boldsymbol{x} = [\boldsymbol{p}^c, (\bar{P}^s, \bar{E}^s, \boldsymbol{p}^s, \boldsymbol{e}^s)_{s\in{1,2}}]$  If `fixed_cap = True`, the storage capacities are either removed from the list of design variables or fixed to their initial values, i.e., $\boldsymbol{x} = [\boldsymbol{p}^c, (\boldsymbol{p}^s, \boldsymbol{e}^s)_{s\in{1,2}}]$.

The objective function of the problem aims to maximize the added NPV, i.e., the contribution of the storage system components to the total NPV,

```{math}
  f_{\text{NPV}, \text{alt}}(\boldsymbol{x}) = \sum_{s} (\lambda^s_P \bar{P}^s + \lambda_E^s \bar{E}^s) - \dfrac{8760}{n}\sum_{k=1}^m {\boldsymbol{\lambda}^T \cdot (\sum_{s} \boldsymbol{p}^{s} - \alpha \boldsymbol{p}^c)}{(1+r)^k}
```
There is only one regularization parameter, $\alpha$, to balance curtailment and storage use. When `fixed_cap = True`, the objective function does not change, but it becomes equivalent to maximization of revenues since the storage capacities are fixed. 

The problem is expressed mathematically as 
```{math}
\begin{align}
\text{min} &&&  f_{\text{NPV}, \text{alt}}(\boldsymbol{x}) \\
\text{s.t.} &&&  P_{\text{bl},i} \leq  p_i \leq \bar{P} &&& i = 0,..., n-1 \\
  &&& p^1_i + p^2_i \leq \text{max}(\bar{P} - \sum_g p^g_i, 0) &&& i = 0,..., n-1 \\
  &&& e^s_0 = e^s_n, &&& s \in \{1,2\} \\
  &&& (1-d)\bar{E} \leq e^{s}_i \leq \bar{E}^s &&& i = 0,..., n \\
  &&& 0 \leq p^c_i \leq \sum_g p_i^g &&& i = 0,..., n-1 \\
  &&& \text{Ramp limit constraints} \\
  &&& \text{Storage model LP-alt} \\
  &&& \text{Storage capacity bounds} 
\end{align}
```

The baseload constraint is described here with a vector $ \boldsymbol{P}_\text{bl}$ and not a scalar, allowing a reliability below 100% to be set. The ramp limit constraint are associated to the parameter `dp_lim` and only enforced if its value is not `None`. The implemented constraints are

$$ -\delta P_\text{lim}  \leq p^{i+1} - p^{i} \leq    \delta P_\text{lim}, \ i = 0, ..., n-2 $$

The optimization problem is described implicitely using the pyomo interface. 



## Rolling horizon dispatch

The function `solve_dispatch` is used for rolling horizon dispatch to simulate the operation of the power plant with imperfect forecast. The goal of this optimization problem is to maximize revenues from electricity sales, while at the same time reducing as much as possible deviations from the required baseload or ramp-limit constraints. In addition, the dispatch problem can be solved to find the best operation considering several scenarios for the power forecast $p^{g,j}, j =1,..., m$.. For simplicity, only one generation component is considered here.

The design variables of the problem are the time series of storage power and energy calculated for each forecast scenario, the time series of curtailed power for each forecast scenario, binary variables indicating if the dispatch constraints are satisfied, and two slack variables $r$ and $r_p$. One has $\boldsymbol{x} = [(\boldsymbol{p}^{c,j})_{j=1,...,m}, (\boldsymbol{p}^{s,j}, \boldsymbol{e}^{s,j})_{s\in{1,2}, j=1,...,m}, \boldsymbol{y}, r, r_p]$ 


The objective function of the problem combines a revenue term, a reliability term using the slack variables and two regularization terms,

```{math}
f(\boldsymbol{x}) = \dfrac{1}{m}\sum_{j=1}^{m} \boldsymbol{\lambda}^T\cdot(\sum_s \boldsymbol{p}^{s,j} - \alpha \boldsymbol{p}^{c,j}) - \mu (r + r_p) + \beta \dfrac{1}{m} \sum_{j=1}^m \sum_s e^s_{n} 
```
where $\mu \gg 1$ and $\beta \ll 1 $. 

The optimization problem can be written mathematically as

```{math}
\begin{align}
\text{max} &&&  f(\boldsymbol{x}) \\
\text{s.t.} &&&  P_\text{bl} y_i \leq p^{g,j}_i + p^{1,j}_i + p^{2,j}_i - p^{c,j}_i \leq \bar{P} &&& i = 0,..., n-1, \ j=1,...,m \\
  &&& p^{1,j}_i + p^{2,j}_i \leq \text{max}(\bar{P} - p^{g,j}_i, 0) &&& i = 0,..., n-1, \ j=1,...,m \\
  &&& e^{s,j}_0 = e^{s,j}_\text{init}, &&& s \in \{1,2\}, \ j=1,...,m \\
  &&& p^{s,0}_0 = p^{s,j}_0 &&& s \in \{1,2\}, \ j=1,...,m \\
  &&& (1-d)\bar{E} \leq e^{s}_i \leq \bar{E}^s &&& i = 0,..., n \\
  &&& 0 \leq p^{c,j}_i \leq p^{g,j}_i &&& i = 0,..., n-1,  \ j=1,...,m \\
  &&& r\geq 0\\
  &&& r_p\geq 0\\
  &&& \text{Reliability constraint} \\
  &&& \text{Ramp limit constraints} \\
  &&& \text{Storage model LP-alt} 
\end{align}
```
where $e^s_\text{init}$ is the initial state-of-charge of storage system $s$ and should match the operation decided at the previous time step.

The reliability constraint enforces a target reliability $r_\text{th}$ based on the number of times where the dispatch constraints (baseload and ramp limit) are satisfied. This calculation is done for the time window of the forecast in addition to a window of past operation of length $n_h$. This enables to keep a *memory* of previous operation. One has

$$ \sum_{i=0}^{n-1} y_i \geq (r_\text{th} - r)(n + n_h) - k_h $$

where $k_h$ is the number of time steps satisfying the dispatch constraints in the window of past operation. The terms $r_\text{th}, n_h$ and $k_h$ are inputs to the function `solve_dispatch_pyomo`. 


The ramp limit constraints are expressed as follows, with a separate expression for the constraint at the first time steps,
```{math}
\begin{align}
 & p^{g,j}_0 + p^{1,j}_0 + p^{2,j}_0 - p^{c,j}_0 - p_h - p^s_h \geq -\delta P_\text{lim} + r_p \bar{P} && j = 1, ...,m \\
 & p^{1,j}_0 + p^{2,j}_0 - p^{c,j}_0 - p_h - p^s_h  \leq y_i ( \delta P_\text{lim} - p^{g,j}_0 + p_h + p^s_h) && j = 1, ...,m \\
& ( p^{1,j}_{i+1} + p^{2,j}_{i+1} - p^{c,j}_{i+1}) - ( p^{1,j}_{i} + p^{2,j}_{i} - p^{c,j}_{i}) \geq -\bar{P} + y_i (\bar{P} - \delta P_\text{lim}) - p^{g,j}_i +  p^{g,j}_{i+1} && j = 1, ...,m, \ i =1, ..., n-1\\
& ( p^{1,j}_{i+1} + p^{2,j}_{i+1} - p^{c,j}_{i+1}) - ( p^{1,j}_{i} + p^{2,j}_{i} - p^{c,j}_{i}) \leq \bar{P} + y_i (-\bar{P} + \delta P_\text{lim}) - p^{g,j}_i +  p^{g,j}_{i+1} && j = 1, ...,m, \ i =1, ..., n-1
\end{align}
```
where $p_{h}$ is the value of the power produced at the previous time step minus curtailment and $p^s_{h}$ the value of the storage power at the previous time step. The lower bound for the ramp-limit at the first time step uses the slack variable $r_p$ instead of a binary variable. This is to reduce the violation of the constraint when the ramp-limit cannot be satisfied at the first time step.

Finally, to facilitate convergence of the algorithm, the bounds on the binary variables are set so that
- in the presence of a baseload constraint, $y_i = 1$ if $\underset{j}{\text{min}}(p^{g,j}_i) \geq P_\text{bl}$
- in the absence of baseload or ramp-limit constraints, $y_i = 1, \forall i$.

The dispatch optimization in `solve_dispatch` is implemented using the `lp-alt` formulation. Two regularization terms are included in the objective function to ensure the constraints are active at the optimum. Furthermore, a penalty on the reliability binary variables is needed to ensure the dispatch constraints are respected as much as possible. As such, the objective function becomes


```{warning}
The rolling horizon dispatch has only been used and verified for one storage system.
```

## Implementation status

The code implement different dispatch optimization problems through three routines. However, not all variations described above are available in each routine. An overview of their differences is reported below.

| Routine name | Formulation | Objective function | Constraints| Optimization algorithm |
| ------------ | ----------- |------------------- |----------- |----------------- |
| `solve_lp_sparse` | lp, lp-alp, milp | NPV or R      | Baseload  | `scipy.linprog` |
| `solve_lp_pyomo`| lp-alt | NPV  or R      | Baseload, Ramp-limit  | pyomo-compatible (mosek, cplex, gurobi, etc.) |
| `solve_dispatch_pyomo`| lp-alt | Trade off between revenues and reliability | Baseload, Ramp-limit      | pyomo-compatible (mosek, cplex, gurobi, etc.) |
