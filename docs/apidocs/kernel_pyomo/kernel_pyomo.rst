:py:mod:`kernel_pyomo`
======================

.. py:module:: kernel_pyomo

.. autodoc2-docstring:: kernel_pyomo
   :parser: docstrings_parser
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`solve_lp_pyomo <kernel_pyomo.solve_lp_pyomo>`
     - .. autodoc2-docstring:: kernel_pyomo.solve_lp_pyomo
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`run_storage_operation <kernel_pyomo.run_storage_operation>`
     - .. autodoc2-docstring:: kernel_pyomo.run_storage_operation
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`solve_dispatch_pyomo <kernel_pyomo.solve_dispatch_pyomo>`
     - .. autodoc2-docstring:: kernel_pyomo.solve_dispatch_pyomo
          :parser: docstrings_parser
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TIME_LIMIT_SHORT <kernel_pyomo.TIME_LIMIT_SHORT>`
     - .. autodoc2-docstring:: kernel_pyomo.TIME_LIMIT_SHORT
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`TIME_LIMIT_LONG <kernel_pyomo.TIME_LIMIT_LONG>`
     - .. autodoc2-docstring:: kernel_pyomo.TIME_LIMIT_LONG
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`DEFAULT_ALPHA_OBJ <kernel_pyomo.DEFAULT_ALPHA_OBJ>`
     - .. autodoc2-docstring:: kernel_pyomo.DEFAULT_ALPHA_OBJ
          :parser: docstrings_parser
          :summary:

API
~~~

.. py:data:: TIME_LIMIT_SHORT
   :canonical: kernel_pyomo.TIME_LIMIT_SHORT
   :value: 2

   .. autodoc2-docstring:: kernel_pyomo.TIME_LIMIT_SHORT
      :parser: docstrings_parser

.. py:data:: TIME_LIMIT_LONG
   :canonical: kernel_pyomo.TIME_LIMIT_LONG
   :value: 180

   .. autodoc2-docstring:: kernel_pyomo.TIME_LIMIT_LONG
      :parser: docstrings_parser

.. py:data:: DEFAULT_ALPHA_OBJ
   :canonical: kernel_pyomo.DEFAULT_ALPHA_OBJ
   :value: None

   .. autodoc2-docstring:: kernel_pyomo.DEFAULT_ALPHA_OBJ
      :parser: docstrings_parser

.. py:function:: solve_lp_pyomo(price_ts: shipp.timeseries.TimeSeries, prod1: shipp.components.Production, prod2: shipp.components.Production, stor1: shipp.components.Storage, stor2: shipp.components.Storage, discount_rate: float, n_year: int, p_min: float, p_max: float, n: int, name_solver: str = 'mosek', fixed_cap: bool = False, dp_lim=None, alpha_obj: float = DEFAULT_ALPHA_OBJ, verbose=False) -> shipp.components.OpSchedule
   :canonical: kernel_pyomo.solve_lp_pyomo

   .. autodoc2-docstring:: kernel_pyomo.solve_lp_pyomo
      :parser: docstrings_parser

.. py:function:: run_storage_operation(run_type: str, power: list, price: list, p_min: float, p_max: float, stor: shipp.components.Storage, e_start: float, n: int, nt: int, dt: float, rel: float = 1.0, forecast: list = None, n_hist: int = 0, verbose: bool = False, name_solver: str = 'mosek', dp_lim=None, beta_obj: float = 1e-06, mu: float = 1.0, alpha_obj: float = DEFAULT_ALPHA_OBJ) -> dict
   :canonical: kernel_pyomo.run_storage_operation

   .. autodoc2-docstring:: kernel_pyomo.run_storage_operation
      :parser: docstrings_parser

.. py:function:: solve_dispatch_pyomo(price: list, m: int, rel: float, n: int, power_forecast: list, p_min: float, p_max: float, e_start1: float, e_start2: float, dt: float, stor1: shipp.components.Storage, stor2: shipp.components.Storage, cnt_hist: int = 0, n_hist: int = 0, verbose: bool = False, name_solver: str = 'mosek', dp_lim: float = None, beta_obj: float = 1e-06, alpha_obj: float = DEFAULT_ALPHA_OBJ, mu: float = 1.0, tol: float = 0.0001, p_hist_res: float = 0, p_hist_stor: float = 0) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, pyomo.environ.SolverStatus]
   :canonical: kernel_pyomo.solve_dispatch_pyomo

   .. autodoc2-docstring:: kernel_pyomo.solve_dispatch_pyomo
      :parser: docstrings_parser
