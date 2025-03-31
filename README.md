# measles-dashboard

## About The Project

This repo contains the code used to run the measles simulator hosted at [https://epiengage-measles.tacc.utexas.edu/](https://epiengage-measles.tacc.utexas.edu/).
The simulator was developed by members of epiENGAGE and TACC(https://tacc.utexas.edu/).

## Getting started

### Running Containerized Dashboard

Start the dashboard:

```docker compose -f docker/docker-compose.yml up --build -d```

Navigate to `localhost:8050` in browser to view.

Stop the dashboard:

```docker compose -f docker/docker-compose.yml down```

### Model and app files

The epidemiological model is implemented in the file `measles_single_population.py`, with model details provided in this pdf(https://epiengage-measles.tacc.utexas.edu/assets/epiENGAGE_Measles_Outbreak_Simulator%E2%80%93Model%20Details-2025.pdf), and at the bottom of the dashboard(https://epiengage-measles.tacc.utexas.edu/).

The rest of the Python files contain the dash app code, with `app.py` being the main file.

The various csv files `XY_MMR_vax_rate.csv` contain MMR vaccination rates at the school or district level for state XY.


## License

Distributed under the BSD 3-Clause license. See `LICENSE` for more information.

## Contact

[utpandemics@austin.utexas.edu](utpandemics@austin.utexas.edu)