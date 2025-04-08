# SHIPP: Sizing optimization for HybrId Power Plants

## Description
SHIPP is used for studying the design and operation of hybrid power plants, i.e. power plants combining one or more renewable energy production with energy storage systems.

SHIPP is in development. Its capabilities are currently limited to sizing and operation of storage systems only. 


## Installation
The package can be installed with pip

```python
pip install shipp
```

## Usage

Examples are given in the folder `examples/`. 

The folder `experiments/` contains scripts to reproduce the results presented in the following publications:
- **`hyb24_bl_hpp/`**: Iori, J., Zaaijer, M., von Terzi, D., & Watson, S. (2024). Design drivers for the storage system of baseload hybrid power plants. In 8th International Hybrid Power Plants and Systems Workshop, HYB 2024 (2 ed., Vol. 2024, pp. 245-250) https://doi.org/10.1049/icp.2024.1844
- **`we25_robust_dispatch/`**: Iori, J., Zaaijer, M., Kreeft, J., von Terzi, D., Watson, S. (2025) _Reliable operation of wind-storage systems for baseload power production_, WindEurope Annual Event 2025 [Accepted for publication] 

## Future developments

- Expand optimization problem definition to an arbitrary number of production and storage objects.
- Include the lifetime of storage systems in the `Storage` objects.
- Remove dependency on class `TimeSeries`

## Dependencies
A valid access or license to a solver compatible with pyomo (MOSEK, CPLEX, Gurobi, etc.) is recommended to solve large problems (see more information here: https://www.pyomo.org/).

## Authors and acknowledgment
This project is developed by Jenna Iori at Delft University of Technology and is part of the Hollandse Kust Noord wind farm innovation program. Funding was provided by CrossWind C.V.

The code is release under the Apache 2.0 License (see License.md).

## Copyright notice: 

Technische Universiteit Delft hereby disclaims all copyright interest in the program “SHIPP” (a design optimization software for hybrid power plants) written by the Author(s). 

Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2024-2025, Jenna Iori