# SHIPP: Sizing optimization for HybrId Power Plants

## Description
SHIPP is used for studying and designing hybrid power plants, i.e. power plants combining one or more renewable energy production with energy storage systems.

SHIPP is in development. Its capabilities are currently limited to sizing and operation of storage systems only. 


## Installation
The package can be installed with pip.

```python
pip install shipp
```

## Usage

An example case is given in the repository folder `examples/`.

The folder `experiments/` contains advanced applications of the code used for scientific conferences and publications. 

Further documentation is available here: https://jennaiori.github.io/shipp/

## Future developments
- Publish package on PyPI
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

© 2024, Jenna Iori