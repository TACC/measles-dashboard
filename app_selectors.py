from dash import Dash, html, dcc, callback, Output, Input, State  # Patch
import dash_bootstrap_components as dbc
import pandas as pd

from app_styles import BASE_FONT_STYLE, \
    SELECTOR_DISPLAY_STYLE, DROPDOWN_BASE_CONFIG, SELECTOR_TOOLTIP_STYLE, \
    NO_WRAP_FULL_WIDTH

# IMPORTANT NOTE: if we change the "min" to 0 or 1 for `I0_selector` and `school_size_selector`,
#   then negative values make the box RED -- but the warning doesn't trigger!
#   It's interesting... the warning only triggers when inputs are received --
#   setting the minimum doesn't allow input reception -- soooo we do NOT set the minimum
#   in the selectors, and we trigger the (external) error message instead!

states_list = ("North Carolina", "Pennsylvania", "Texas")

# Ew I don't like this... but time-sensitive deadline means we
#   gotta change this later...
# This is for Texas
county_options_default_list = list(pd.read_csv("Texas_Counties.csv").columns)
school_options_default_list = list(pd.read_csv("Texas_Travis_Schools.csv").columns)

SELECTOR_DEFAULTS =\
    {"county_selector_default": "",
     "state_selector_default": "",
     "school_selector_default": "",
     "county_options_default": [],
     "school_options_default": [],
     "school_size": 500,
     "vax_rate": 85,
     "I0": 1,
     "R0": 15.0,
     "latent_period": 10.5,
     "infectious_period": 5.0,
     "outbreak_threshold": 10,
     "vaccine_efficacy_selector": 0.997,
     "vaccinated_infectiousness_selector": 0.05,
     }

school_size_selector = dcc.Input(
    id='school_size_selector',
    type='number',
    placeholder='School enrollment (number of students)',
    value=SELECTOR_DEFAULTS["school_size"],
    debounce=False,
    style={**SELECTOR_DISPLAY_STYLE, **BASE_FONT_STYLE}
)

I0_selector = dcc.Input(
    id='I0_selector',
    type='number',
    placeholder='Number of students initially infected',
    value=SELECTOR_DEFAULTS["I0"],
    debounce=False,
    style={**SELECTOR_DISPLAY_STYLE, 'margin-left': 'auto', **BASE_FONT_STYLE}
)

vaccination_rate_selector = dcc.Input(
    id='vax_rate_selector',
    type='number',
    placeholder='Vaccination rate (%)',
    value=SELECTOR_DEFAULTS["vax_rate"],
    style={**SELECTOR_DISPLAY_STYLE, **BASE_FONT_STYLE,
           'width': '7ch'}
)

state_selector = html.Div(
    [
        dbc.Label("Select State"),
        dcc.Dropdown(
            id="state_selector",
            options=states_list,
            value=SELECTOR_DEFAULTS["state_selector_default"],
            **DROPDOWN_BASE_CONFIG
        ),
    ], className="mb-4",
    style={**BASE_FONT_STYLE}
)

county_selector = html.Div(
    [
        dbc.Label("Select County", html_for="county_selector"),
        dcc.Dropdown(
            id="county_selector",
            options=SELECTOR_DEFAULTS["county_options_default"],
            value=SELECTOR_DEFAULTS["county_selector_default"],
            **DROPDOWN_BASE_CONFIG,
            style={**NO_WRAP_FULL_WIDTH},

        ),
    ], className="mb-4 m-0",
    style={**BASE_FONT_STYLE, 'whiteSpace': 'nowrap', 'overflow': 'visible'}
)

school_selector = html.Div(
    [
        dbc.Label("Select School/District", html_for="school_selector",
                  style={**BASE_FONT_STYLE}),
        dcc.Dropdown(
            id="school_selector",
            options=SELECTOR_DEFAULTS["school_options_default"],
            value=SELECTOR_DEFAULTS["school_selector_default"],
            **DROPDOWN_BASE_CONFIG,
            style={**NO_WRAP_FULL_WIDTH, 'font-size': '14pt'},
        ),
    ], className="mb-4",
    style={**BASE_FONT_STYLE, 'whiteSpace': 'normal', 'width': '100%'}
)

R0_selector = dcc.Slider(
    id='R0_selector',
    min=12,
    max=18,
    step=0.1,
    value=SELECTOR_DEFAULTS["R0"],
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
    value=SELECTOR_DEFAULTS["latent_period"],
    included=False,
    marks={7: {'label': '7', 'style': {**BASE_FONT_STYLE}},
           10.5: {'label': '10.5', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
           12: {'label': '12', 'style': {**BASE_FONT_STYLE}},
           },
    tooltip={**SELECTOR_TOOLTIP_STYLE},
)

infectious_period_selector = dcc.Slider(
    id='infectious_period_selector',
    min=4,
    max=9,
    step=0.1,
    value=SELECTOR_DEFAULTS["infectious_period"],
    included=False,
    marks={4: {'label': '4', 'style': {**BASE_FONT_STYLE}},
           5: {'label': '5', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
           9: {'label': '9', 'style': {**BASE_FONT_STYLE}}
           },
    tooltip={**SELECTOR_TOOLTIP_STYLE},
)

threshold_selector = dcc.Slider(
    id='threshold_selector',
    min=3,
    max=25,
    step=1,
    value=SELECTOR_DEFAULTS["outbreak_threshold"],
    included=False,
    marks={3: {'label': '3', 'style': {**BASE_FONT_STYLE}},
           10: {'label': '10', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
           25: {'label': '25', 'style': {**BASE_FONT_STYLE}}
           },
    tooltip={**SELECTOR_TOOLTIP_STYLE},
)

vaccine_efficacy_selector = dcc.Slider(
    id='vaccine_efficacy_selector',
    min=0.98,
    max=1.0,
    step=0.001,
    value=SELECTOR_DEFAULTS["vaccine_efficacy_selector"],
    included=False,
    marks={
        0.98: {'label': '0.98', 'style': {**BASE_FONT_STYLE}},
        0.997: {'label': '0.997', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
        1: {'label': '1.0', 'style': {**BASE_FONT_STYLE}},
        },
    tooltip={**SELECTOR_TOOLTIP_STYLE},
)

vaccinated_infectiousness_selector = dcc.Slider(
    id='vaccinated_infectiousness_selector',
    min=0.0,
    max=0.5,
    step=0.01,
    value=SELECTOR_DEFAULTS["vaccinated_infectiousness_selector"],
    included=False,
    marks={
        0: {'label': '0.0', 'style': {**BASE_FONT_STYLE}},
        0.05: {'label': '0.05', 'style': {**BASE_FONT_STYLE, 'fontWeight': 'bold'}},
        0.5: {'label': '0.5', 'style': {**BASE_FONT_STYLE}},
        },
    tooltip={**SELECTOR_TOOLTIP_STYLE},
)