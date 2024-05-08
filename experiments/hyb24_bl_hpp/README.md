# Design optimization of baseload hybrid power plants

## Description

This folder contains the code used for the publication "Design drivers for baseload hybrid power plants" submitted to the 2024 International Hybrid power plants and systems workshop (https://hybridpowersystems.org/). 

## Dependencies and requirements
In order to run the code, the following is required:
- an active token for renewables.ninja, available with for registered users (see instructions here: https://www.renewables.ninja/documentation/api)
- an active token for the API of the entso-e transparency platform (see instructions here: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
- a license to an optimization software that is compatible with pyomo, such as mosek, gurobi, cplex (see more information here: https://www.pyomo.org/)

## Usage
The jupyter notebook main.ipynb is used to run all numerical experiments and generate plots. The notebook relies on an input file describing the site locations (site2.csv with 2 sites or site.csv with 30 sites). Note that solving all the optimization problems for the 30 sites can take around 4 hours on a standard laptop 


## Authors and acknowledgment
This project is developed by Jenna Iori at Delft University of Technology and is part of the Hollandse Kust Noord wind farm innovation program. Funding was provided by CrossWind C.V.
