#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:39:51 2025
Updated on Thu Feb 20 1:55:00 2025

@author: rfp437
"""
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

INPUT_DEFAULTS = {
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
    id='vax_rate',
    type='number',
    placeholder='Vaccination rate (%)',
    value=85,
    min=0,
    max=100,
    style={**SELECTOR_DISPLAY_STYLE, **BASE_FONT_STYLE,
           'width': '7ch'}
)

school_size_selector = dcc.Input(
    id='school_size',
    type='number',
    placeholder='School enrollment (number of students)',
    value=500,
    debounce=False,
    style={**SELECTOR_DISPLAY_STYLE, **BASE_FONT_STYLE}
)

I0_selector = dcc.Input(
    id='I0',
    type='number',
    placeholder='Number of students initially infected',
    value=1.0,
    min=0,
    debounce=False,
    style={**SELECTOR_DISPLAY_STYLE, 'margin-left': 'auto', **BASE_FONT_STYLE}
)

R0_selector = dcc.Slider(
    id='R0',
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
    id='latent_period',
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
    id='infectious_period',
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


def check_inputs_validity(init_infected: int,
                          total_enrollment: int,
                          vax_proportion: float):
    """
    IMPORTANT: vax_proportion must be between [0,1] --

    TODO: fix inconsistencies with variable naming for vax percent --
        I can see this being a cause of a bug/misunderstanding in the future
        -- sometimes it's in percent form (so like an int, like 95)
        and sometimes it's in decimal form
    """
    if total_enrollment is None or init_infected is None or vax_proportion is None:
        return False

    elif init_infected > int((1 - vax_proportion) * total_enrollment):
        return False

    else:
        return True


@callback(
    [Output('spaghetti_plot', 'figure'),
     Output('prob_20plus_new_str', 'children'),
     Output('cases_expected_over_20_str', 'children'),
     Output('warning_str', 'children')
     ],
    [Input('school_size', 'value'),
     Input('vax_rate', 'value'),
     Input('I0', 'value'),
     Input('R0', 'value'),
     Input('latent_period', 'value'),
     Input('infectious_period', 'value')]
)
def update_graph(school_size,
                 vax_rate,
                 I0,
                 R0,
                 latent_period,
                 infectious_period):

    school_size = school_size if school_size is not None else INPUT_DEFAULTS['school_size']
    vax_rate = vax_rate if vax_rate is not None else INPUT_DEFAULTS['vax_rate']
    I0 = I0 if I0 is not None else INPUT_DEFAULTS['I0']
    R0 = R0 if R0 is not None else INPUT_DEFAULTS['R0']
    latent_period = latent_period if latent_period is not None else INPUT_DEFAULTS['latent_period']
    infectious_period = infectious_period if infectious_period is not None else INPUT_DEFAULTS['infectious_period']

    R0 = max(R0, 0)

    # Update parameters, run simulations
    n_sim = DASHBOARD_CONFIG["num_simulations"]

    params = copy.deepcopy(msp.MSP_PARAMS)
    params['population'] = [int(school_size)]
    params['vaccinated_percent'] = [0.01 * float(vax_rate)]
    params['I0'] = [int(I0)]
    params['R0'] = float(R0)
    params['incubation_period'] = float(latent_period)
    params['infectious_period'] = float(infectious_period)

    inputs_are_valid = check_inputs_validity(init_infected=params["I0"][0],
                                             total_enrollment=params["population"][0],
                                             vax_proportion=params["vaccinated_percent"][0])

    if inputs_are_valid:
        stochastic_sim = msp.StochasticSimulations(
            params, n_sim, print_summary_stats=False, show_plots=False)

        fig = msp.gimme_spaghetti_infected_ma(sim=stochastic_sim,
                                              nb_curves_displayed=20,
                                              curve_selection_seed=DASHBOARD_CONFIG["spaghetti_curve_selection_seed"])

        prob_20plus_new_str, cases_expected_over_20_str = \
            msp.create_strs_20plus_new_and_outbreak(stochastic_sim,
                                                    DASHBOARD_CONFIG["outbreak_size_uncertainty_displayed"])

        return fig, prob_20plus_new_str, cases_expected_over_20_str, ""

    else:
        warning_str = "Invalid inputs: there are more initially " \
                  "infected than unvaccinated students. Please adjust."

        return px.line(), "", "", warning_str


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
    Output('vax_rate', 'value'),
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
    app.run(debug=True, host='0.0.0.0')