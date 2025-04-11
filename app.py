#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rfp437

>(^_^)> ~~~~
"""

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
import measles_efficiency
import subprocess

from randomgen import PCG64

from app_static_graphics import navbar, bottom_info_section, \
    bottom_credits, school_outbreak_projections_header
from app_dynamic_graphics import results_header, spaghetti_plot_section, \
    dashboard_input_panel
from app_computation_functions import dashboard_results_fig, \
    dashboard_spaghetti, EMPTY_SPAGHETTI_PLOT_INFECTED_MA, \
    dashboard_new_cases_cond_mean_str, dashboard_exceedance_prob_str,\
    dashboard_percentiles_str
from app_selectors import SELECTOR_DEFAULTS

DASHBOARD_CONFIG = {
    'num_simulations_graph': 20,
    'num_simulations_results': 1000,
    'simulation_seed': 147125098488,
    'spaghetti_curve_selection_seed': 12345,
}
STATE_DATA_SUBFOLDER = 'state_data/'

msp.DEFAULT_MSP_PARAMS["simulation_seed"] = DASHBOARD_CONFIG["simulation_seed"]

# TODO -- again, this can be streamlined... but here is
#   the initial stab at generalizing the code to multiple states...
#   We can also cache the dataframe subsets...
# TODO -- maybe... put state-to-CSV mapping in JSON file?

state_name_to_two_letter_code = {
    "Connecticut": "CT",
    "Maryland": "MD",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "Pennsylvania": "PA",
    "Texas": "TX",
}


state_to_df_map = {
    state_name: pd.read_csv(STATE_DATA_SUBFOLDER + state_code + '_MMR_vax_rate.csv')
    for state_name, state_code in state_name_to_two_letter_code.items()
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
     Input('infectious_period_selector', 'value'),
     Input('threshold_selector', 'value'),
     Input('vaccine_efficacy_selector', 'value'),
     Input('vaccinated_infectiousness_selector', 'value')]
)
def create_params_from_selectors(params_dict,
                                 school_size,
                                 vax_rate_percent,
                                 I0,
                                 R0,
                                 latent_period,
                                 infectious_period,
                                 outbreak_threshold,
                                 vaccine_efficacy,
                                 vaccinated_infectiousness):
    school_size = school_size if school_size is not None else SELECTOR_DEFAULTS['school_size']
    vax_rate_percent = vax_rate_percent if vax_rate_percent is not None else SELECTOR_DEFAULTS['vax_rate']
    I0 = I0 if I0 is not None else SELECTOR_DEFAULTS['I0']
    R0 = R0 if R0 is not None else SELECTOR_DEFAULTS['R0']
    latent_period = latent_period if latent_period is not None else SELECTOR_DEFAULTS['latent_period']
    infectious_period = infectious_period if infectious_period is not None else SELECTOR_DEFAULTS[
        'infectious_period']
    outbreak_threshold = outbreak_threshold if outbreak_threshold is not None else SELECTOR_DEFAULTS[
        'outbreak_threshold']
    vaccine_efficacy = vaccine_efficacy if vaccine_efficacy is not None else SELECTOR_DEFAULTS[
        'vaccine_efficacy_selector']
    vaccinated_infectiousness = vaccinated_infectiousness if vaccinated_infectiousness is not None else \
        SELECTOR_DEFAULTS[
            'vaccinated_infectiousness_selector']

    params_dict['population'] = [int(school_size)]
    params_dict['vax_prop'] = [0.01 * float(vax_rate_percent)]
    params_dict['I0'] = [int(I0)]
    params_dict['R0'] = float(R0)
    params_dict['incubation_period'] = float(latent_period)
    params_dict['infectious_period'] = float(infectious_period)
    params_dict['threshold_values'] = [int(outbreak_threshold)]
    params_dict['vaccine_efficacy'] = float(0.01 * vaccine_efficacy)
    params_dict['relative_infectiousness_vaccinated'] = float(1 - 0.01 * vaccinated_infectiousness)

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
    # Assuming single population -- again, single population / multiple population stuff
    #   is confusing here -- and the hardcoding could accidentally lead to mistakes in future

    if not 0 <= params_dict["vax_prop"][0] <= 1:
        warning_str = "Invalid inputs: vaccination rate must be between 0-100%."
        return False, warning_str
    elif params_dict["I0"][0] < 0 or params_dict["population"][0] < 0:
        warning_str = "Invalid inputs: school enrollment and students initially " \
                      "infected must be positive whole numbers. Please adjust."
        return False, warning_str
    elif params_dict["I0"][0] > int((1 - params_dict["vax_prop"][0]) * params_dict["population"][0]):
        warning_str = "Invalid inputs: The number of initially infected students " \
                      "cannot exceed the number of unvaccinated students. Please adjust."
        return False, warning_str
    else:
        return True, ""


@callback(
    [Output('spaghetti_plot', 'figure'),
     Output('prob_threshold_plus_new_str', 'children'),
     Output('cases_expected_over_threshold_str', 'children'),
     Output('cases_expected_over_threshold_vaccinated_str', 'children'),
     Output('outbreak_title', 'children'),
     Output('cases_condition', 'children')],
    [Input('dashboard_params', 'data'),
     Input('inputs_are_valid', 'data')]
)
def update_graph(params_dict: dict,
                 inputs_are_valid: bool):

    """
    TODO: I am sorry Remy, I have made this function grotesque
    -- will split this up into functions later!
    """

    n_sim_graph = DASHBOARD_CONFIG["num_simulations_graph"]

    if inputs_are_valid:

        I_unvax_init = params_dict["I0"][0]

        threshold_val = int(params_dict['threshold_values'][0])
        outbreak_title_str = 'Chance of exceeding {} new infections'.format(threshold_val)
        cases_condition_str = '*if exceeds {} new infections*'.format(threshold_val)

        measles_graph_results = msp.DashboardExperiment(params_dict, n_sim_graph)
        measles_graph_results.ma7_num_infected.create_df_simple_spaghetti()

        transition_sampler = measles_efficiency.build_transition_sampler(
            np.random.Generator(PCG64(seed=DASHBOARD_CONFIG["simulation_seed"])))

        unvax_cases_array, vax_cases_array = \
            measles_efficiency.compute_new_infections(params_dict,
                                                      transition_sampler,
                                                      DASHBOARD_CONFIG["num_simulations_results"])

        total_cases_array = unvax_cases_array + vax_cases_array
        total_cases_above_threshold_array = total_cases_array[total_cases_array >= threshold_val]

        prob_threshold_plus_new_str = \
            dashboard_exceedance_prob_str(len(total_cases_above_threshold_array) / DASHBOARD_CONFIG["num_simulations_results"])

        if len(total_cases_above_threshold_array) == 0:
            cases_expected_over_threshold_unvaccinated_str = "Fewer than {} new infections".format(
                int(threshold_val))
            cases_expected_over_threshold_breakthrough_str = ""
        else:
            unvax_cases_lb, unvax_cases_ub = np.percentile(unvax_cases_array[total_cases_array >= threshold_val], [2.5, 97.5])
            vax_cases_lb, vax_cases_ub = np.percentile(vax_cases_array[total_cases_array >= threshold_val], [2.5, 97.5])
            cases_expected_over_threshold_unvaccinated_str = \
                'Unvaccinated cases: ' + dashboard_percentiles_str(I_unvax_init, unvax_cases_lb, unvax_cases_ub)
            cases_expected_over_threshold_breakthrough_str = \
                'Vaccinated cases: ' + dashboard_percentiles_str(0, vax_cases_lb, vax_cases_ub)

        ix_median = np.argmin(np.abs(measles_graph_results.total_new_cases.data - np.median(unvax_cases_array + vax_cases_array)))

        fig = dashboard_results_fig(df_spaghetti=measles_graph_results.ma7_num_infected.df_simple_spaghetti,
                                    index_sim_closest_median=ix_median,
                                    nb_curves_displayed=20,
                                    curve_selection_seed=DASHBOARD_CONFIG[
                                            "spaghetti_curve_selection_seed"])

        return fig, prob_threshold_plus_new_str, cases_expected_over_threshold_unvaccinated_str, \
               cases_expected_over_threshold_breakthrough_str, \
               outbreak_title_str, cases_condition_str

    else:

        # Note -- returning None instead of empty dict is a big mistake --
        #   doesn't work and also messes up the graphs for correct inputs!
        #   Be very careful with the syntax here.
        return EMPTY_SPAGHETTI_PLOT_INFECTED_MA, "", "", "", "", ""


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
        f"{name} ({age_group})" if pd.notna(age_group) and age_group != "" else f"{name}"
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

        # Handle cases where age_group is present or ""
        if ' (' in school_with_age_str:
            school, age_group = school_with_age_str.split(' (')
            age_group = age_group.rstrip(")")
        else:
            school = school_with_age_str
            age_group = ""

        # Filter DataFrame based on the presence or absence of age_group
        df_school = county_subset_df.loc[
            (county_subset_df['School District or Name'] == school) &
            ((county_subset_df['Age Group'] == age_group) | (
                    pd.isna(county_subset_df['Age Group']) & (age_group == "")))
            ]

        school_vax_rate_pct = df_school['MMR Vaccination Rate'].values[0]

        return [school_vax_rate_pct]

    else:

        return [SELECTOR_DEFAULTS["vax_rate"]]


result = subprocess.run("git symbolic-ref -q --short HEAD || git describe --tags --exact-match",
                        shell=True, capture_output=True)
version = result.stdout.decode("utf-8").strip() if result.stdout else "Unknown"

app = Dash(
    prevent_initial_callbacks='initial_duplicate')
server = app.server  # Do we need this?
app.title = f"epiENGAGE Measles Outbreak Simulator v-{version}"

# Add inline script to initialize Google Analytics
app.scripts.append_script({
    'external_url': 'https://www.googletagmanager.com/gtag/js?id=G-QS2CT3051Y'
})
app.scripts.append_script({'external_url': '/assets/gtag.js'})

app.layout = dbc.Container(
    [
        dcc.Store(id="inputs_are_valid", data=True),
        dcc.Store(id="dashboard_params", data=copy.deepcopy(msp.DEFAULT_MSP_PARAMS)),

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
    app.run(debug=False, host='0.0.0.0')
