#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:39:51 2025
Updated on Thu Feb 20 1:55:00 2025

@author: rfp437
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
import subprocess

from enum import Enum

from app_static_graphics import navbar, footer, \
    vaccination_rate_label, school_size_label, I0_label, R0_label, \
    latent_period_label, infectious_period_label, bottom_info_section, \
    bottom_credits, school_outbreak_projections_header
from app_dynamic_graphics import results_header, spaghetti_plot_section, \
    inputs_panels
from app_styles import BASE_FONT_STYLE, BASE_FONT_FAMILY_STR, \
    SELECTOR_DISPLAY_STYLE, DROPDOWN_BASE_CONFIG, SELECTOR_TOOLTIP_STYLE, \
    SELECTOR_NOTE_STYLE, RESULTS_HEADER_STYLE

DASHBOARD_CONFIG = {
    'num_simulations': 200,
    'outbreak_size_uncertainty_displayed': msp.OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY_FIVE,
    'simulation_seed': 147125098488,
    'spaghetti_curve_selection_seed': 12345
}

msp.MSP_PARAMS["simulation_seed"] = DASHBOARD_CONFIG["simulation_seed"]

DASHBOARD_INPUT_DEFAULTS = {
    'school_size': 500,
    'vax_rate': 0.0,
    'I0': 1,
    'R0': 15.0,
    'latent_period': 10.5,
    'infectious_period': 8.0}

# TODO: move selectors to their own file
# TODO: what is df and df_county doing? The way these dataframes are
#   hardcoded into these functions seems risky.

df = pd.read_csv('TX_MMR_vax_rate.csv')

initial_county = 'Travis'
states = ["Texas"]

state_dropdown = html.Div(
    [
        dbc.Label("Select State"),
        dcc.Dropdown(
            id="state-dropdown",
            options=states,
            value="Texas",
            **DROPDOWN_BASE_CONFIG
        ),
    ], className="mb-4",
    style={**BASE_FONT_STYLE}
)

county_dropdown = html.Div(
    [
        dbc.Label("Select Texas County", html_for="county_dropdown"),
        dcc.Dropdown(
            id="county-dropdown",
            options=sorted(df["County"].unique()),
            value=initial_county,
            **DROPDOWN_BASE_CONFIG,
            style={"whiteSpace": "nowrap", "width": "100%"},

        ),
    ], className="mb-4 m-0",
    style={**BASE_FONT_STYLE, 'whiteSpace': 'nowrap', 'overflow': 'visible'}
)

# df there should depend on the selected county
df_county = df.loc[df['County'] == initial_county]

school_options = sorted(
    f"{name} ({age_group})"
    for name, age_group in zip(df_county["School District or Name"], df_county["age_group"])
)

initial_school = 'AUSTIN ISD (Kindergarten)'

if initial_school not in school_options:
    initial_school = school_options[0]

school_dropdown = html.Div(
    [
        dbc.Label("Select a School District", html_for="school_dropdown",
                  style={**BASE_FONT_STYLE}),
        dcc.Dropdown(
            id="school-dropdown",
            options=school_options,
            value=initial_school,
            **DROPDOWN_BASE_CONFIG,
            style={"whiteSpace": "nowrap", "width": "100%", 'font-size': '14pt'},
        ),
    ], className="mb-4",
    style={**BASE_FONT_STYLE, 'whiteSpace': 'normal', 'width': '100%'}
)

vaccination_rate_selector = dcc.Input(
    id='vax_rate_selector',
    type='number',
    placeholder='Vaccination rate (%)',
    value=85,
    min=0,
    max=100,
    style={**SELECTOR_DISPLAY_STYLE, **BASE_FONT_STYLE,
           'width': '7ch'}
)

school_size_selector = dcc.Input(
    id='school_size_selector',
    type='number',
    placeholder='School enrollment (number of students)',
    value=500,
    debounce=False,
    style={**SELECTOR_DISPLAY_STYLE, **BASE_FONT_STYLE}
)

I0_selector = dcc.Input(
    id='I0_selector',
    type='number',
    placeholder='Number of students initially infected',
    value=1.0,
    min=0,
    debounce=False,
    style={**SELECTOR_DISPLAY_STYLE, 'margin-left': 'auto', **BASE_FONT_STYLE}
)

R0_selector = dcc.Slider(
    id='R0_selector',
    min=12,
    max=18,
    step=0.1,
    value=15,
    included=False,
    marks={12: {'label': '12', 'style': {**BASE_FONT_STYLE}},
           15: {'label': '15', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
           18: {'label': '18', 'style': {**BASE_FONT_STYLE}}
           },
    tooltip={**SELECTOR_TOOLTIP_STYLE},

)

latent_period_selector = dcc.Slider(
    id='latent_period_selector',
    min=7,
    max=12,
    step=0.1,
    value=10.5,
    included=False,
    marks={7: {'label': '7', 'style': {**BASE_FONT_STYLE}},
           10.5: {'label': '10.5', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
           12: {'label': '12', 'style': {**BASE_FONT_STYLE}},
           },
    tooltip={**SELECTOR_TOOLTIP_STYLE},
)

infectious_period_selector = dcc.Slider(
    id='infectious_period_selector',
    min=5,
    max=9,
    step=0.1,
    value=8,
    included=False,
    marks={5: {'label': '5', 'style': {**BASE_FONT_STYLE}},
           8: {'label': '8', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
           9: {'label': '9', 'style': {**BASE_FONT_STYLE}}
           },
    tooltip={**SELECTOR_TOOLTIP_STYLE},
)

result = subprocess.run("git symbolic-ref -q --short HEAD || git describe --tags --exact-match",
                        shell=True, capture_output=True)
version = result.stdout.decode("utf-8").strip() if result.stdout else "Unknown"

app = Dash(
    prevent_initial_callbacks='initial_duplicate')
server = app.server
app.title = f"epiENGAGE Measles Outbreak Simulator v-{version}"

# Add inline script to initialize Google Analytics
app.scripts.append_script({
    'external_url': 'https://www.googletagmanager.com/gtag/js?id=G-QS2CT3051Y'
})
app.scripts.append_script({'external_url': '/assets/gtag.js'})

# Define the accordion separately
epi_params_accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                dbc.Col(
                    [
                        html.Br(),
                        dbc.Col(html.Div(R0_label), className="mb-2"),
                        dbc.Col(html.Div(R0_selector), className="mb-2"),
                        dbc.Col(html.Div(latent_period_label), className="mb-2"),
                        dbc.Col(html.Div(latent_period_selector), className="mb-2"),
                        dbc.Col(html.Div(infectious_period_label), className="mb-2"),
                        dbc.Col(html.Div(infectious_period_selector), className="mb-2"),
                    ]
                ),
                title="Change Parameters ▾",
                style={"textAlign": "center"}  # Add this line to center the text
            ),
        ],
        flush=True,
        always_open=False,  # Ensures sections can be toggled independently
        active_item=[],  # Empty list means all sections are closed by default
    )
)

# Define the accordion separately
school_district_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            html.Div(
                [
                    html.H3("ISD rates are district averages.", style={**SELECTOR_NOTE_STYLE}),
                    html.H3("Rates at individual schools may be higher or lower.", style={**SELECTOR_NOTE_STYLE}),
                    dbc.Col(html.Div(state_dropdown), className="mb-2 p-0"),
                    dbc.Col(html.Div(county_dropdown), className="mb-2 p-0"),
                    dbc.Col(html.Div(school_dropdown), className="mb-2 p-0"),
                ]
            ),
            title="School/District Lookup ▾ ",
            style={"font-size": "18pt", "width": "100%", "margin": "none"},
            className="m-0"
        ),
    ],
    flush=True,
    always_open=False,  # Ensures sections can be toggled independently
    active_item=[],  # Empty list means all sections are closed by default
)


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

            inputs_panels(school_size_header=school_size_label,
                          school_size_input=school_size_selector,
                          I0_header=I0_label,
                          I0_input=I0_selector,
                          vaccination_rate_header=vaccination_rate_label,
                          vaccination_rate_input=vaccination_rate_selector,
                          top_accordion=school_district_accordion,
                          bottom_accordion=epi_params_accordion),
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
                                 vax_rate,
                                 I0,
                                 R0,
                                 latent_period,
                                 infectious_period):

    school_size = school_size if school_size is not None else DASHBOARD_INPUT_DEFAULTS['school_size']
    vax_rate = vax_rate if vax_rate is not None else DASHBOARD_INPUT_DEFAULTS['vax_rate']
    I0 = I0 if I0 is not None else DASHBOARD_INPUT_DEFAULTS['I0']
    R0 = R0 if R0 is not None else DASHBOARD_INPUT_DEFAULTS['R0']
    latent_period = latent_period if latent_period is not None else DASHBOARD_INPUT_DEFAULTS['latent_period']
    infectious_period = infectious_period if infectious_period is not None else DASHBOARD_INPUT_DEFAULTS['infectious_period']

    params_dict['population'] = [int(school_size)]
    params_dict['vaccinated_percent'] = [0.01 * float(vax_rate)]
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

    if params_dict["I0"][0] > int((1 - params_dict["vaccinated_percent"][0]) * params_dict["population"][0]):
        return False, warning_str

    else:
        return True, ""


@callback(
    [Output('spaghetti_plot', 'figure'),
     Output('prob_20plus_new_str', 'children'),
     Output('cases_expected_over_20_str', 'children')
     ],
    [Input('dashboard_params', 'data'),
     Input('inputs_are_valid', 'data')]
)
def update_graph(params_dict: dict,
                 inputs_are_valid: bool):

    # Update parameters, run simulations
    n_sim = DASHBOARD_CONFIG["num_simulations"]

    if inputs_are_valid:
        stochastic_sim = msp.StochasticSimulations(
            params_dict, n_sim, print_summary_stats=False, show_plots=False)

        fig = msp.gimme_spaghetti_infected_ma(sim=stochastic_sim,
                                              nb_curves_displayed=20,
                                              curve_selection_seed=DASHBOARD_CONFIG["spaghetti_curve_selection_seed"])

        prob_20plus_new_str, cases_expected_over_20_str = \
            msp.create_strs_20plus_new_and_outbreak(stochastic_sim,
                                                    DASHBOARD_CONFIG["outbreak_size_uncertainty_displayed"])

        return fig, prob_20plus_new_str, cases_expected_over_20_str

    else:

        # Note -- returning None instead of empty dict is a big mistake --
        #   doesn't work and also messes up the graphs for correct inputs!
        #   Be very careful with the syntax here.
        return {}, "", ""


@callback(
    [Output('school-dropdown', 'options'),
     Output('school-dropdown', 'value')  # ,
     ],
    [Input('county-dropdown', 'value')],
    prevent_initial_call=True
)
def update_school_selector(county):
    df_county = df.loc[df['County'] == county]
    new_school_options = sorted(
        f"{name} ({age_group})"
        for name, age_group in zip(df_county["School District or Name"], df_county["age_group"])
    )
    school_selected = new_school_options[0]

    return new_school_options, school_selected


@callback(
    Output('vax_rate_selector', 'value'),
    [Input('school-dropdown', 'value')]
)
def update_school_vax_rate(school_with_age):  # county):
    school, age_group = school_with_age.split(' (')
    age_group = age_group.rstrip(")")

    df_school = df.loc[
        # (df['County'] == county) &
        (df['School District or Name'] == school) &
        (df['age_group'] == age_group)
        ]

    if not df_school.empty:
        school_vax_rate_pct = df_school['MMR_Vaccination_Rate'].values[0]
        school_vax_rate = float(school_vax_rate_pct.replace('%', ''))
        return school_vax_rate
    else:
        school_vax_rate = 85


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')