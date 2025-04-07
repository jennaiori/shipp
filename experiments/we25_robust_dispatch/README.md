# Robust Dispatch Optimization

This folder contains experiments related to robust dispatch optimization for wind-storage systems. These files allows to reproduce the results published in the following article: Iori, J., Zaaijer, M., Kreeft, J., von Terzi, D., Watson, S. (2025) _Reliable operation of wind-storage systems for baseload power production_, WindEurope Annual Event 2025 [Accepted for publication] 

## Files

- **`aux_forecast.py`**: Module containing auxiliary functions.
- **`compute_power_forecast.py`**: Script for creating the windpower forecast and observation data files.
- **`create_price_files.py`**: Script for creating price data input files, using the ENTSO-E Rest API
- **`create_comparison_input_files.py`**: Script for creating the input files used for running the numerical experiments.
- **`run_comparison.py`**: Script for running the numerical experiments.
- **`postprocess_one_site.py`**: Post-processing script for the results of one wind farm site.
- **`postprocess_all_sites.py`**: Post-processing script to compare the results of all wind farm sites.
- **`data/`** and **`results/`**: Directory for storing the input data and the results
    - **`data/sites.csv`**: CSV file linking site name, site location and associated electricity market
    - **`data/windpower_parameters.json`** and **`data/forecast_parameters.json`**: Files containing parameters for generating the wind power observation and forecast data 


## How to Run

1. Ensure all dependencies are installed:
    ```bash
    pip install shipp
    pip install <name_of_optimization_solver_package>
    ```
2. Download the input data (see Resources below) in the folder `data/`
3. Write your ENTSO-E token in a new text file `data/token_entsoe.txt`
4. (Optional) Change the name of the optimization solver in the files `run_comparison.py` and `create_comparison_input_files.py`
5. Create the price files and wind power files

```bash
SITES=("hkn" "bor" "hkz" "gem" "god" "vej" "hr3" "anh" "kfl" "bea" "sea" "mor" "tri" "ark" "stb" "stn" "hs1" "hs2")

for SITE in "${SITES[@]}"; do
        python compute_power_forecast.py data/ data/ data/windpower_parameters.json data/forecast_parameters.json $SITE 0
        python compute_power_forecast.py data/ data/ data/windpower_parameters.json data/forecast_parameters.json $SITE 1
        python compute_power_forecast.py data/ data/ data/windpower_parameters.json data/forecast_parameters.json $SITE 2

done

python create_price_files.py data/ data/sites.csv data/token_entsoe.txt
```
6. Create input files
```bash
for SITE in "${SITES[@]}"; do
        python create_comparison_input_files.py "data/" "results/" "data/sites.csv" "$SITE"
done
```
7. Run the numerical experiments
```bash
for SITE in "${SITES[@]}"; do

        DIR_OUT="results/res_$SITE"

        mkdir -p $DIR_OUT

        echo "Running optimization for site $SITE"
        python run_comparison.py "results/dispatch_input_file_$SITE.json" "$DIR_OUT/"

done
```

8. Run the postprocessing scripts
```bash
python postprocess_one_site.py
python postprocess_all_sites.py
```

## Resources

The data required to run the scripts is available open-source:  Iori, Jenna (2025): Data associated with the publication "Reliable operation of wind-storage systems for baseload power production". Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/52056a89-9324-486f-a16b-93c0a1ceae2a.v1

## Notes

- Modify configuration parameters in the scripts as needed.
- Results will be saved in the `results/` directory.

## Authors and acknowledgment
This project is developed by Jenna Iori at Delft University of Technology and is part of the Hollandse Kust Noord wind farm innovation program. Funding was provided by CrossWind C.V. 

The code is release under the Apache 2.0 License.