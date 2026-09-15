# SHIPP: Sizing optimization for HybrId Power Plants

[![CI/CD test suite](https://github.com/jennaiori/shipp/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/jennaiori/shipp/actions/workflows/main.yml)
[![DOI](https://img.shields.io/badge/DOI-10.4121%2F2EE36148--369F--4E1F--B770--C86752D7DCA4-yellow.svg)](https://doi.org/10.4121/2EE36148-369F-4E1F-B770-C86752D7DCA4)
[![PyPi](https://img.shields.io/pypi/v/shipp)](https://pypi.org/project/shipp/)
[![License](https://img.shields.io/pypi/l/shipp)](https://github.com/jennaiori/shipp/blob/main/LICENSE)


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/shipp_extended_dark_small.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/shipp_extended_small.png">
  <img alt="SHIPP logo">
</picture>

## Description
SHIPP is used for studying the design and operation of hybrid power plants, i.e. power plants combining one or more renewable energy production with energy storage systems.

Documentation is available at [https://jennaiori.github.io/shipp/](https://jennaiori.github.io/shipp/)

## Installation and Usage
The package can be installed using pip.

```python
pip install shipp
```

Examples are given in the folder `examples/`. 


## Dependencies
A valid access or license to a solver compatible with pyomo (MOSEK, CPLEX, Gurobi, etc.) is recommended to solve large problems (see more information here: https://www.pyomo.org/).

## Latest changes [1.2.2]
- Changed class `Storage`: the depth-of-charge parameter `dod` is replaced by a minimum and maximum state-of-charge `soc_min` and `soc_max`.
- Changed functions `solve_lp_pyomo`, `solve_lp_sparse` and `solve_dispatch_pyomo`: the input `p_min` is now optional and default to 0.
- Changed function `solve_dispatch_pyomo` to take a single dictionary argument `options` for optional arguments like `fixed_cap` and the penalty factors (similar to  `solve_lp_pyomo` and `solve_lp_sparse`).
- Changed optimization problem implemented in `solve_dispatch_pyomo`
    - The ramp-limit is now characterized by separate up- and down- bounds `dp_min` and `dp_max`, with `dp_min` $\leq 0$.
    - Addition and renaming of tuning parameters `beta_obj` and `gamma_obj` in alignment with the optimization problems implemented in `solve_lp_sparse` and `solve_lp_pyomo`.
    - Addition of a slack variable to reduce deviations to the baseload constraint, similar to what was implemented for the ramp constraint.
    - Reformulation of the ramp-limit and baseload constraint at the first time step to combine a binary variable and a slack variable. 
    - Addition of tuning parameters `mu1_obj`, `mu2_obj` and `mu3_obj` for the total reliability and the slack variables of the baseload and ramp penalties, respectively.

## Authors and acknowledgment
This project is developed by Jenna Iori at Delft University of Technology and was initially part of the Hollandse Kust Noord wind farm innovation program, with funding from CrossWind C.V.

The code is release under the Apache 2.0 License (see License.md).

## Copyright notice: 

Technische Universiteit Delft hereby disclaims all copyright interest in the program “SHIPP” (a design optimization software for hybrid power plants) written by the Author(s). 

Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2024-2026, Jenna Iori
