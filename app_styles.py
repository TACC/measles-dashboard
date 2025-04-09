from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

NO_WRAP_FULL_WIDTH = {
    "whiteSpace": "nowrap",
    "width": "100%"
}

DROPDOWN_BASE_CONFIG = {
    "clearable": False,
    "maxHeight": 600,
    "optionHeight": 50
}

BASE_FONT_FAMILY_STR = '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif'

BASE_FONT_STYLE = {
    "fontFamily": "Sans-serif",
    "font-size": "15pt",
}

SELECTOR_DISPLAY_STYLE = {
    'display': 'flex',
    'flexDirection': 'column',
    'textAlign': 'center',
    'width': '6ch'
}

SELECTOR_LABEL_STYLE = {'display': 'inline-block', 'margin-right': 5, 'margin-left': 5, 'fontFamily': 'Sans-serif',
           'font-size': '16pt'}

SELECTOR_TOOLTIP_STYLE = {'placement': 'top', 'always_visible': True, "style": {"fontSize": "16pt"}}

SELECTOR_NOTE_STYLE = {"text-align": "center",
                       "font-family": BASE_FONT_FAMILY_STR,
                       "font-size": "12pt",
                       "font-weight": "400",
                       "font-style": "italic",
                       "line-height": "1"}

RESULTS_HEADER_STYLE = {'color': '#black', 'fontWeight': '500',
                        'font-size': '20pt', "margin": "none"}

SPAGHETTI_PLOT_AXIS_CONFIG = {
    'showgrid': True,
    'gridcolor': "rgb(242,242,242)",
    'title_font': {
        'size': 20,
        'color': "black",
        'family': "Sans-serif"
    },
    'tickfont': {
        'size': 16,
        'color': "black",
        'family': "Sans-serif"
    },
    'zeroline': True,
    'zerolinecolor': "white",
    'linewidth': 2,
    'mirror': True,
    'range': [0, None]
}

BOTTOM_CREDITS_TEXT_STYLE = {
    "color": "#ffffff", "text-decoration": "none", "font-size": "11px"
}