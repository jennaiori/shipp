:py:mod:`kernel`
================

.. py:module:: kernel

.. autodoc2-docstring:: kernel
   :parser: docstrings_parser
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`build_lp_obj_npv <kernel.build_lp_obj_npv>`
     - .. autodoc2-docstring:: kernel.build_lp_obj_npv
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`build_lp_cst_sparse <kernel.build_lp_cst_sparse>`
     - .. autodoc2-docstring:: kernel.build_lp_cst_sparse
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`solve_lp_sparse <kernel.solve_lp_sparse>`
     - .. autodoc2-docstring:: kernel.solve_lp_sparse
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`os_rule_based <kernel.os_rule_based>`
     - .. autodoc2-docstring:: kernel.os_rule_based
          :parser: docstrings_parser
          :summary:

API
~~~

.. py:function:: build_lp_obj_npv(price: numpy.ndarray, n: int, batt_p_cost: float, batt_e_cost: float, h2_p_cost: float, h2_e_cost: float, discount_rate: float, n_year: int) -> numpy.ndarray
   :canonical: kernel.build_lp_obj_npv

   .. autodoc2-docstring:: kernel.build_lp_obj_npv
      :parser: docstrings_parser

.. py:function:: build_lp_cst_sparse(power: numpy.ndarray, dt: float, p_min, p_max: float, n: int, eps_batt: float, eps_h2: float, rate_batt: float = -1.0, rate_h2: float = -1.0, max_soc: float = -1.0, max_h2: float = -1.0, fixed_cap=False) -> tuple[scipy.sparse.coo_matrix, numpy.ndarray, scipy.sparse.coo_matrix, numpy.ndarray, numpy.ndarray, numpy.ndarray]
   :canonical: kernel.build_lp_cst_sparse

   .. autodoc2-docstring:: kernel.build_lp_cst_sparse
      :parser: docstrings_parser

.. py:function:: solve_lp_sparse(price_ts: shipp.timeseries.TimeSeries, prod_wind: shipp.components.Production, prod_pv: shipp.components.Production, stor_batt: shipp.components.Storage, stor_h2: shipp.components.Storage, discount_rate: float, n_year: int, p_min, p_max: float, n: int, fixed_cap: bool = False) -> shipp.components.OpSchedule
   :canonical: kernel.solve_lp_sparse

   .. autodoc2-docstring:: kernel.solve_lp_sparse
      :parser: docstrings_parser

.. py:function:: os_rule_based(price_ts: shipp.timeseries.TimeSeries, prod_wind: shipp.components.Production, prod_pv: shipp.components.Production, stor_batt: shipp.components.Storage, stor_h2: shipp.components.Storage, discount_rate: float, n_year: int, p_min, p_rule: float, price_min: float, n: int, e_start: float = 0) -> shipp.components.OpSchedule
   :canonical: kernel.os_rule_based

   .. autodoc2-docstring:: kernel.os_rule_based
      :parser: docstrings_parser
