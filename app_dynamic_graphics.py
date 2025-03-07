# These functions/components CHANGE based on callbacks
#   -- contains results from simulations based on user inputs

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

from app_styles import BASE_FONT_FAMILY_STR, SELECTOR_NOTE_STYLE, RESULTS_HEADER_STYLE


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
                                dcc.Graph(id="spaghetti_plot"),
                            ]),
                            style={'border': 'none', 'padding': '0'},
                        ),
                    ),
                ], style={"border-top": "2px solid black", "border-left": "1em", "padding": "none", "height": "60%",
                          "width": "100%", "margin-top": "1em"})


def inputs_panels(school_size_header: html.H4,
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
                            html.H3("Enter value or select from Lookup.", style={**SELECTOR_NOTE_STYLE}),
                            html.H3("Update School Enrollment above.", style={**SELECTOR_NOTE_STYLE}),
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