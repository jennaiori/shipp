# Changelog

## [Unreleased]
- Change output of `run_storage_dispatch` to an `OpSchedule` object
- Add degradation model
- Change NPV formulation in the dispatch optimization problem to include OPEX and storage replacement


## [1.2.1] - 2026-06-24
- Added CHANGELOG.md (this document)
- Added documentation on the implementation of the dispatch optimization problems and mathematical background: [https://jennaiori.github.io/shipp/](https://jennaiori.github.io/shipp/)
- Added API reference in the documentation
- Added function `financial_metrics` in `kernel.py` to compute LCOE, NPV, IRR, CAPEX and cashflow
- Changed functions `solve_lp_pyomo` and `solve_lp_sparse`:
    - Both functions now take a single dictionary argument `options` for optional arguments like `fixed_cap` and the penalty factors.
    - The constraint on the first and last state-of-charge is now an inequality constraint $e_0 \leq e_{n+1}$
    - Addition of the constraint on maximum combined storage power in `solve_lp_sparse` to match `solve_lp_pyomo`
    - Addition of a penalty on total curtailed energy with factor `beta_obj`
- Added two problem formulations (`lp` and `milp`) in `solve_lp_sparse`
- Added function `curtail` in `Production` object to copy the object and curtail the power production by a given `np.ndarray`
- Added tests to check conservation of energy between energy produced, curtailed energy, storage losses and energy delivered
- Fixed [issue #4](https://github.com/jennaiori/shipp/issues/4)
- Fixed lack of storage model check / storage losses check in `solve_lp_sparse`


## [1.2.0] - 2026-02-23
- Updated documentation: [https://jennaiori.github.io/shipp/](https://jennaiori.github.io/shipp/)
- Added penalty factor `alpha_obj` in `run_storage_operation`
- Update of classes `Storage` and `OpSchedule`: addition of input validity checks, function to calculate the minimum allowed state-of-charge, separation of revenues and annual revenues
- Removed `solve_dispatch_pyomo_windonly`
- Added depth of discharge (`dod`) as a property of `Storage` 
- Added ramp-limitation constraint in `solve_lp_pyomo` and `run_storage_dispatch`
- Addition of curtailment as a design variable of the dispatch problems


## [1.1] - 2025-04-07
- Updated documentation: [https://jennaiori.github.io/shipp/](https://jennaiori.github.io/shipp/)
- Added function `run_storage_dispatch` for online dispatch optimization with power forecast

## [1.0] - 2024-10-09
- Inital version of code SHIPP