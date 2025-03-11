# These functions/components CHANGE based on callbacks
#   -- contains results from simulations based on user inputs

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

from app_static_graphics import \
    vaccination_rate_label, school_size_label, I0_label, R0_label, \
    latent_period_label, infectious_period_label
from app_styles import BASE_FONT_FAMILY_STR, RESULTS_HEADER_STYLE, \
    SELECTOR_NOTE_STYLE
from app_computation_functions import EMPTY_SPAGHETTI_PLOT_INFECTED_MA
from app_selectors import school_size_selector, \
    I0_selector, vaccination_rate_selector, state_selector, \
    county_selector, school_selector, R0_selector, latent_period_selector, \
    infectious_period_selector


def results_header():
    return dbc.Row(
        [
            # Chance of an Outbreak
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(id='outbreak',
                                                 children='Chance of exceeding 20 new infections',
                                                 style={**RESULTS_HEADER_STYLE, 'fontWeight': '500'}
                                                 ),
                                    dcc.Markdown(id='prob_20plus_new_str',
                                                 style={**RESULTS_HEADER_STYLE, 'color': '#bf5700',
                                                        "font-size": "22pt", "font-weight": "800"}
                                                 ),
                                ],
                                style={
                                    'textAlign': 'center',
                                    'fontFamily': BASE_FONT_FAMILY_STR,
                                    'fontSize': '18pt',
                                    'border': 'none'
                                }
                            )
                        ]
                    ),
                    style={'border': 'none'}
                ),
            ),

            # Expected Outbreak Size
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(id='cases',
                                                 children='Likely outbreak size',
                                                 style={'color': '#black', 'fontWeight': '500',
                                                        'font-size': '20pt', 'margin': 'none'}
                                                 ),
                                    dcc.Markdown("*if exceeds 20 new infections*",
                                                 style={'font-size': '14pt', "margin": "none"}),
                                    dcc.Markdown(id='cases_expected_over_20_str',
                                                 style={'color': '#bf5700', 'fontWeight': '800',
                                                        'font-size': '22pt', 'margin-top': '0.5em'}
                                                 ),
                                ],
                                style={
                                    'textAlign': 'center',
                                    'fontFamily': BASE_FONT_FAMILY_STR,
                                    'fontSize': '18pt',
                                    'border': 'none'
                                }
                            )
                        ]
                    ),
                    style={'border': 'none'}
                ),
                style={'borderLeft': '3px solid #bf5700'}
            ),
        ],
    )


def spaghetti_plot_section():
    return dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3("This graph shows 20 plausible school outbreak curves.",
                            style={"text-align": "center", "margin-top": "1em", "margin-bottom": "1em",
                                   "margin-left": "1.8em",
                                   "font-family": BASE_FONT_FAMILY_STR,
                                   "font-size": "14pt", "font-weight": "400", "font-style": "italic"}),
                    dcc.Graph(id="spaghetti_plot", figure=EMPTY_SPAGHETTI_PLOT_INFECTED_MA),
                ]),
                style={'border': 'none', 'padding': '0'},
            ),
        ),
    ], style={"border-top": "2px solid black", "border-left": "1em", "padding": "none", "height": "60%",
              "width": "100%", "margin-top": "1em"})


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
                    html.H3("School enrollment doesn't change automatically.",
                            style={**SELECTOR_NOTE_STYLE, "font-weight": "bold"}),
                    html.H3("Please update the value manually.", style={**SELECTOR_NOTE_STYLE, "font-weight": "bold"}),
                    dbc.Col(html.Div(state_selector), className="mb-2 p-0"),
                    dbc.Col(html.Div(county_selector), className="mb-2 p-0"),
                    dbc.Col(html.Div(school_selector), className="mb-2 p-0"),
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


def input_panel_builder(school_size_header: html.H4,
                        school_size_input: dcc.Input,
                        I0_header: html.H4,
                        I0_input: dcc.Input,
                        vaccination_rate_header: html.H4,
                        vaccination_rate_input: dcc.Input,
                        top_accordion: dbc.Accordion,
                        bottom_accordion: dbc.Accordion) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H3("Model Inputs", style={"margin-left": "0.2em", "margin-top": "0.5em",
                                                   "font-family": BASE_FONT_FAMILY_STR,
                                                   "font-size": "24pt", "font-weight": "500",
                                                   "textAlign": "center"}, className="mt-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='warning_str',
                                     style={"color": "red", "font-size": "12", "text-align": "center"},
                                     className="d-flex flex-column align-items-center")
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div(school_size_header),
                            html.Div(school_size_input),
                        ], className="d-flex flex-column align-items-center"),
                    ], className="d-flex flex-column align-items-center mb-2"),

                    dbc.Row([
                        dbc.Col([
                            html.Div(I0_header),
                            html.Div(I0_input)
                        ], className="d-flex flex-column align-items-center"),
                    ], className="d-flex flex-column align-items-center mb-2"),

                    dbc.Row([
                        dbc.Col(html.Div(vaccination_rate_header),
                                className="d-flex flex-column align-items-center"),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.H3("Enter value or select from Lookup.", style={**SELECTOR_NOTE_STYLE})
                        ]),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.Div(vaccination_rate_input), html.Div(" OR ", style={"font-size": "16pt",
                                                                                      "margin-top": "0.5em",
                                                                                      "margin-bottom": "0.5em"}),
                            html.Div(top_accordion, style={"width": "100%", "textAlign": "center"}),
                        ], className="d-flex flex-column align-items-center"),
                    ], style={"border-bottom": "2px solid black", "margin-right": "0.2em"}),

                    html.Br(),
                    html.H3("Epidemic Parameters", style={"margin-left": "0.2em", "margin-top": "0.5em",
                                                          "font-family": BASE_FONT_FAMILY_STR,
                                                          "font-size": "24pt", "font-weight": "500",
                                                          "textAlign": "center"}),
                    html.Br(),
                    dbc.Row(dbc.Col(html.I(
                        "Caution – Default values reflect published estimates. Significant changes may result in inaccurate projections."),
                        className="mb-2 align-items-center",
                        style={"font-size": "14pt", "textAlign": "center"})),

                    dbc.Row([
                        dbc.Col(bottom_accordion, className="mb-2", style={"width": "100%", "textAlign": "center"}),
                    ]),
                ]
            ),
            style={'border': 'none'}
        ),
        width=3, xs=12, sm=12, md=12, lg=12, xl=3,
        style={"border-right": "2px solid black", "padding": "10px"},
    )


def dashboard_input_panel():
    return input_panel_builder(school_size_header=school_size_label,
                               school_size_input=school_size_selector,
                               I0_header=I0_label,
                               I0_input=I0_selector,
                               vaccination_rate_header=vaccination_rate_label,
                               vaccination_rate_input=vaccination_rate_selector,
                               top_accordion=school_district_accordion,
                               bottom_accordion=epi_params_accordion)
