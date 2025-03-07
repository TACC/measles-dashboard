from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

DROPDOWN_BASE_CONFIG = {
    "clearable": False,
    "maxHeight": 600,
    "optionHeight": 50
}

BASE_FONT_FAMILY_STR = '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif'

BASE_FONT_STYLE = {
    "fontFamily": "Sans-serif",
    "font-size": "16pt",
}

SELECTOR_DISPLAY_STYLE = {
    'display': 'flex',
    'flexDirection': 'column',
    'textAlign': 'center',
    'width': '6ch'
}

SELECTOR_TOOLTIP_STYLE = {'placement': 'top', 'always_visible': True, "style": {"fontSize": "16pt"}}

SELECTOR_NOTE_STYLE = {"text-align": "center",
                       "font-family": BASE_FONT_FAMILY_STR,
                       "font-size": "12pt",
                       "font-weight": "400",
                       "font-style": "italic",
                       "line-height": "1"}

RESULTS_HEADER_STYLE = {'color': '#black', 'fontWeight': '500',
                        'font-size': '20pt', "margin": "none"}