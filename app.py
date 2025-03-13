#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rfp437

>(^_^)> ~~~~
"""

# TODO: we should probably make the dictionary of params be a
#   dataclass with a decent name like MSP_params or something -- because
#   we keep passing it as an argument and ASSUME certain things about the field names
#   and data types -- so we should make these assumptions explicit, and the
#   dataclass does that for us

# TODO: streamline the single population / multiple population stuff?
#   Some confusing stuff... e.g. remembering that some parameters are actually
#   a LIST (but of 1 element, for 1 subpopulation)

from dash import Dash, html, dcc, callback, Output, Input, State  # Patch
import plotly.express as px
import pandas as pd
import numpy as np
import copy
import dash_bootstrap_components as dbc
import measles_single_population as msp
import subprocess

from app_static_graphics import navbar, bottom_info_section, \
    bottom_credits, school_outbreak_projections_header
from app_dynamic_graphics import results_header, spaghetti_plot_section, \
    dashboard_input_panel
from app_computation_functions import create_data_spaghetti_plot_infected_ma, \
    get_spaghetti_plot_infected_ma, EMPTY_SPAGHETTI_PLOT_INFECTED_MA
from app_selectors import SELECTOR_DEFAULTS

DASHBOARD_CONFIG = {
    'num_simulations': 200,
    'outbreak_size_uncertainty_displayed': msp.OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY_FIVE,
    'simulation_seed': 147125098488,
    'spaghetti_curve_selection_seed': 12345,
}

msp.MSP_PARAMS["simulation_seed"] = DASHBOARD_CONFIG["simulation_seed"]

# TODO -- again, this can be streamlined... but here is
#   the initial stab at generalizing the code to multiple states...
#   We can also cache the dataframe subsets...
# TODO -- maybe... put state-to-CSV mapping in JSON file?

TX_df = pd.read_csv('TX_MMR_vax_rate.csv')
NC_df = pd.read_csv('NC_MMR_vax_rate.csv')

states = ("Texas", "North Carolina")

state_to_df_map = {
    "Texas": TX_df,
    "North Carolina": NC_df
}


def get_county_subset_df(state_str, county_str) -> pd.DataFrame:

    state_subset_df = state_to_df_map[state_str]
    county_subset_df = state_subset_df[state_subset_df["County"] == county_str]

    return county_subset_df


# Generalizing state/county/school selection for different states
#   After state selection, county selection options should update
#   After county selection, school selection options should update


@callback(
    [Output('dashboard_params', 'data')],
    [State('dashboard_params', 'data'),
     Input('school_size_selector', 'value'),
     Input('vax_rate_selector', 'value'),
     Input('I0_selector', 'value'),
     Input('R0_selector', 'value'),
     Input('latent_period_selector', 'value'),
     Input('infectious_period_selector', 'value')])
def create_params_from_selectors(params_dict,
                                 school_size,
                                 vax_rate_percent,
                                 I0,
                                 R0,
                                 latent_period,
                                 infectious_period):
    school_size = school_size if school_size is not None else SELECTOR_DEFAULTS['school_size']
    vax_rate_percent = vax_rate_percent if vax_rate_percent is not None else SELECTOR_DEFAULTS['vax_rate_percent']
    I0 = I0 if I0 is not None else SELECTOR_DEFAULTS['I0']
    R0 = R0 if R0 is not None else SELECTOR_DEFAULTS['R0']
    latent_period = latent_period if latent_period is not None else SELECTOR_DEFAULTS['latent_period']
    infectious_period = infectious_period if infectious_period is not None else SELECTOR_DEFAULTS[
        'infectious_period']

    params_dict['population'] = [int(school_size)]
    params_dict['vax_prop'] = [0.01 * float(vax_rate_percent)]
    params_dict['I0'] = [int(I0)]
    params_dict['R0'] = float(R0)
    params_dict['incubation_period'] = float(latent_period)
    params_dict['infectious_period'] = float(infectious_period)

    # Bug I got stuck on for awhile -- dcc.State can certainly handle dictionaries
    # HOWEVER -- callbacks always expect the return type to be a list or tuple
    #   if there are multiple values -- so we wrap the dictionary in a list,
    #   but we do not have to modify anything else -- dash just knows how to
    #   parse this output :)
    return [params_dict]


@callback(
    [Output('inputs_are_valid', 'data'),
     Output('warning_str', 'children')],
    [Input('dashboard_params', 'data')]
)
def check_inputs_validity(params_dict: dict) -> str:
    """
    IMPORTANT: vax_proportion must be between [0,1] --

    TODO: fix inconsistencies with variable naming for vax percent --
        I can see this being a cause of a bug/misunderstanding in the future
        -- sometimes it's in percent form (so like an int, like 95)
        and sometimes it's in decimal form
    """

    warning_str = "Invalid inputs: The number of initially infected students " \
                  "cannot exceed the number of unvaccinated students. Please adjust."

    # Assuming single population -- again, single population / multiple population stuff
    #   is confusing here -- and the hardcoding could accidentally lead to mistakes in future

    if params_dict["I0"][0] > int((1 - params_dict["vax_prop"][0]) * params_dict["population"][0]):
        return False, warning_str

    else:
        return True, ""


@callback(
    [Output('spaghetti_plot', 'figure'),
     Output('prob_20plus_new_str', 'children'),
     Output('cases_expected_over_20_str', 'children')],
    [Input('dashboard_params', 'data'),
     Input('inputs_are_valid', 'data')]
)
def update_graph(params_dict: dict,
                 inputs_are_valid: bool):
    # Update parameters, run simulations
    n_sim = DASHBOARD_CONFIG["num_simulations"]

    if inputs_are_valid:
        stochastic_sim = msp.StochasticSimulations(params_dict, n_sim)
        stochastic_sim.run_stochastic_model(track_infected=True)
        stochastic_sim.prep_across_rep_plot_data(include_infected_7day_ma=True)
        stochastic_sim.calculate_20_plus_new_cases_statistics()

        prob_20plus_new_str, cases_expected_over_20_str = \
            stochastic_sim.create_strs_20plus_new_and_outbreak(DASHBOARD_CONFIG["outbreak_size_uncertainty_displayed"])

        (plot_df, plot_color_map) = \
            create_data_spaghetti_plot_infected_ma(sim=stochastic_sim,
                                                   nb_curves_displayed=20,
                                                   curve_selection_seed=DASHBOARD_CONFIG[
                                                       "spaghetti_curve_selection_seed"])

        fig = get_spaghetti_plot_infected_ma(plot_df, plot_color_map)

        # ">" needs an escape in HTML!!!!
        if prob_20plus_new_str == "> 99%":
            prob_20plus_new_str = "\> 99%"

        return fig, prob_20plus_new_str, cases_expected_over_20_str

    else:

        # Note -- returning None instead of empty dict is a big mistake --
        #   doesn't work and also messes up the graphs for correct inputs!
        #   Be very careful with the syntax here.
        return EMPTY_SPAGHETTI_PLOT_INFECTED_MA, "", ""


@callback(
    [Output("county_selector", "options"),
     Output("county_selector", "value")],
    [Input("state_selector", "value")],
    prevent_initial_call=True
)
def update_county_selector(state):

    new_county_options = sorted(state_to_df_map[state]["County"].unique())
    default_county_displayed = new_county_options[0]

    return new_county_options, default_county_displayed



@callback(
    [Output('school_selector', 'options'),
     Output('school_selector', 'value')],
    [State('state_selector', 'value'),
     Input('county_selector', 'value')],
    prevent_initial_call=True
)
def update_school_selector(state, county):

    df = get_county_subset_df(state, county)
    new_school_options = sorted(
        f"{name} ({age_group})"
        for name, age_group in zip(df["School District or Name"], df["Age Group"])
    )
    default_school_displayed = new_school_options[0]

    return new_school_options, default_school_displayed


@callback(
    [Output('vax_rate_selector', 'value')],
    [State('state_selector', 'value'),
     State('county_selector', 'value'),
     Input('school_selector', 'value')],
    prevent_initial_call=True
)
def get_school_vax_rate(state_str,
                        county_str,
                        school_with_age_str) -> float:

    if school_with_age_str:

        county_subset_df = get_county_subset_df(state_str, county_str)

        school, age_group = school_with_age_str.split(' (')
        age_group = age_group.rstrip(")")

        df_school = county_subset_df.loc[
            (county_subset_df['School District or Name'] == school) &
            (county_subset_df['Age Group'] == age_group)]

        school_vax_rate_pct = df_school['MMR Vaccination Rate'].values[0]

        return [school_vax_rate_pct]

    else:

        return [SELECTOR_DEFAULTS["vax_rate"]]


result = subprocess.run("git symbolic-ref -q --short HEAD || git describe --tags --exact-match",
                        shell=True, capture_output=True)
version = result.stdout.decode("utf-8").strip() if result.stdout else "Unknown"

app = Dash(
    prevent_initial_callbacks='initial_duplicate')
server = app.server     # Do we need this?
app.title = f"epiENGAGE Measles Outbreak Simulator v-{version}"

# Add inline script to initialize Google Analytics
app.scripts.append_script({
    'external_url': 'https://www.googletagmanager.com/gtag/js?id=G-QS2CT3051Y'
})
app.scripts.append_script({'external_url': '/assets/gtag.js'})

app.layout = dbc.Container(
    [
        dcc.Store(id="inputs_are_valid", data=True),
        dcc.Store(id="dashboard_params", data=copy.deepcopy(msp.MSP_PARAMS)),

        dbc.Row([navbar], className="my-2"),
        html.Br(),
        html.Br(),

        # Main Layout with Left and Right Sections
        dbc.Row([
            # Left section
            html.Br(),

            dashboard_input_panel(),

            dbc.Col([
                html.Br(),

                school_outbreak_projections_header(),

                html.Br(),

                results_header(),

                html.Br(),

                spaghetti_plot_section()], className="col-xl-9")
        ]),

        html.Br(),

        bottom_info_section(),

        bottom_credits()
    ], fluid=True, style={"min-height": "100vh", "display": "flex", "flex-direction": "column"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
