:py:mod:`components`
====================

.. py:module:: components

.. autodoc2-docstring:: components
   :parser: docstrings_parser
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Storage <components.Storage>`
     - .. autodoc2-docstring:: components.Storage
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`Production <components.Production>`
     - .. autodoc2-docstring:: components.Production
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`OpSchedule <components.OpSchedule>`
     - .. autodoc2-docstring:: components.OpSchedule
          :parser: docstrings_parser
          :summary:

API
~~~

.. py:class:: Storage(e_cap: float = 0, p_cap: float = 0, eff_in: float = 1, eff_out: float = 1, e_cost: float = 0, p_cost: float = 0, dod: float = 1)
   :canonical: components.Storage

   .. autodoc2-docstring:: components.Storage
      :parser: docstrings_parser

   .. rubric:: Initialization

   .. autodoc2-docstring:: components.Storage.__init__
      :parser: docstrings_parser

   .. py:method:: get_av_eff() -> float
      :canonical: components.Storage.get_av_eff

      .. autodoc2-docstring:: components.Storage.get_av_eff
         :parser: docstrings_parser

   .. py:method:: get_rt_eff() -> float
      :canonical: components.Storage.get_rt_eff

      .. autodoc2-docstring:: components.Storage.get_rt_eff
         :parser: docstrings_parser

   .. py:method:: get_tot_costs() -> float
      :canonical: components.Storage.get_tot_costs

      .. autodoc2-docstring:: components.Storage.get_tot_costs
         :parser: docstrings_parser

   .. py:method:: get_min_e() -> float
      :canonical: components.Storage.get_min_e

      .. autodoc2-docstring:: components.Storage.get_min_e
         :parser: docstrings_parser

   .. py:method:: copy() -> components.Storage
      :canonical: components.Storage.copy

      .. autodoc2-docstring:: components.Storage.copy
         :parser: docstrings_parser

   .. py:method:: __repr__() -> str
      :canonical: components.Storage.__repr__

.. py:class:: Production(power_ts: shipp.timeseries.TimeSeries, p_cost: float = 0)
   :canonical: components.Production

   .. autodoc2-docstring:: components.Production
      :parser: docstrings_parser

   .. rubric:: Initialization

   .. autodoc2-docstring:: components.Production.__init__
      :parser: docstrings_parser

   .. py:method:: get_tot_costs() -> float
      :canonical: components.Production.get_tot_costs

      .. autodoc2-docstring:: components.Production.get_tot_costs
         :parser: docstrings_parser

   .. py:method:: __repr__() -> str
      :canonical: components.Production.__repr__

.. py:class:: OpSchedule(production_list: list[components.Production], storage_list: list[components.Storage], production_p: list[shipp.timeseries.TimeSeries], storage_p: list[shipp.timeseries.TimeSeries], storage_e: list[shipp.timeseries.TimeSeries], price: numpy.ndarray = None)
   :canonical: components.OpSchedule

   .. autodoc2-docstring:: components.OpSchedule
      :parser: docstrings_parser

   .. rubric:: Initialization

   .. autodoc2-docstring:: components.OpSchedule.__init__
      :parser: docstrings_parser

   .. py:method:: update_capex() -> None
      :canonical: components.OpSchedule.update_capex

      .. autodoc2-docstring:: components.OpSchedule.update_capex
         :parser: docstrings_parser

   .. py:method:: update_revenue(price: numpy.ndarray) -> None
      :canonical: components.OpSchedule.update_revenue

      .. autodoc2-docstring:: components.OpSchedule.update_revenue
         :parser: docstrings_parser

   .. py:method:: get_npv_irr(discount_rate: float, n_year: int) -> tuple[float, float]
      :canonical: components.OpSchedule.get_npv_irr

      .. autodoc2-docstring:: components.OpSchedule.get_npv_irr
         :parser: docstrings_parser

   .. py:method:: get_added_npv(discount_rate: float, n_year: int) -> float
      :canonical: components.OpSchedule.get_added_npv

      .. autodoc2-docstring:: components.OpSchedule.get_added_npv
         :parser: docstrings_parser

   .. py:method:: get_power_partition() -> list[float]
      :canonical: components.OpSchedule.get_power_partition

      .. autodoc2-docstring:: components.OpSchedule.get_power_partition
         :parser: docstrings_parser

   .. py:method:: check_losses(tol: float, verbose: bool = False) -> bool
      :canonical: components.OpSchedule.check_losses

      .. autodoc2-docstring:: components.OpSchedule.check_losses
         :parser: docstrings_parser

   .. py:method:: plot_powerflow(label_list: list[str] = None, xlabel: str = 'Time [day]', ylabel1: str = 'Power [MW]', ylabel2: str = 'Energy [MWh]') -> None
      :canonical: components.OpSchedule.plot_powerflow

      .. autodoc2-docstring:: components.OpSchedule.plot_powerflow
         :parser: docstrings_parser

   .. py:method:: plot_powerout(label_list: list[str] = None, xlabel: str = 'Time [day]', ylabel: str = 'Power [MW]', xlim: list[float] = None) -> None
      :canonical: components.OpSchedule.plot_powerout

      .. autodoc2-docstring:: components.OpSchedule.plot_powerout
         :parser: docstrings_parser
