:py:mod:`io_functions`
======================

.. py:module:: io_functions

.. autodoc2-docstring:: io_functions
   :parser: docstrings_parser
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`replace_nan_with_mean <io_functions.replace_nan_with_mean>`
     - .. autodoc2-docstring:: io_functions.replace_nan_with_mean
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`api_request_rninja <io_functions.api_request_rninja>`
     - .. autodoc2-docstring:: io_functions.api_request_rninja
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`api_request_entsoe <io_functions.api_request_entsoe>`
     - .. autodoc2-docstring:: io_functions.api_request_entsoe
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`get_power_price_data <io_functions.get_power_price_data>`
     - .. autodoc2-docstring:: io_functions.get_power_price_data
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`save_power_price_to_json <io_functions.save_power_price_to_json>`
     - .. autodoc2-docstring:: io_functions.save_power_price_to_json
          :parser: docstrings_parser
          :summary:
   * - :py:obj:`get_power_price_from_json <io_functions.get_power_price_from_json>`
     - .. autodoc2-docstring:: io_functions.get_power_price_from_json
          :parser: docstrings_parser
          :summary:

API
~~~

.. py:function:: replace_nan_with_mean(data: numpy.ndarray) -> numpy.ndarray
   :canonical: io_functions.replace_nan_with_mean

   .. autodoc2-docstring:: io_functions.replace_nan_with_mean
      :parser: docstrings_parser

.. py:function:: api_request_rninja(token: str, latitude: float, longitude: float, date_start: str, date_end: str, capacity: float = 8000, height: float = 164, turbine: str = 'Vestas V164 8000') -> tuple[numpy.ndarray, numpy.ndarray]
   :canonical: io_functions.api_request_rninja

   .. autodoc2-docstring:: io_functions.api_request_rninja
      :parser: docstrings_parser

.. py:function:: api_request_entsoe(token: str, date_start: str, date_end: str, country_code: str = 'NL') -> numpy.ndarray
   :canonical: io_functions.api_request_entsoe

   .. autodoc2-docstring:: io_functions.api_request_entsoe
      :parser: docstrings_parser

.. py:function:: get_power_price_data(token_rninja: str, token_entsoe: str, date_start: str, date_end: str, latitude: float, longitude: float, capacity: float = 8000, height: float = 164, turbine: str = 'Vestas V164 8000', country_code: str = 'NL') -> tuple[numpy.ndarray, numpy.ndarray]
   :canonical: io_functions.get_power_price_data

   .. autodoc2-docstring:: io_functions.get_power_price_data
      :parser: docstrings_parser

.. py:function:: save_power_price_to_json(filename: str, data_power: numpy.ndarray, data_price: numpy.ndarray)
   :canonical: io_functions.save_power_price_to_json

   .. autodoc2-docstring:: io_functions.save_power_price_to_json
      :parser: docstrings_parser

.. py:function:: get_power_price_from_json(filename: str) -> tuple[numpy.ndarray, numpy.ndarray]
   :canonical: io_functions.get_power_price_from_json

   .. autodoc2-docstring:: io_functions.get_power_price_from_json
      :parser: docstrings_parser
