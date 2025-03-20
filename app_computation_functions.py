import measles_single_population as msp
import pandas as pd
import numpy as np
import plotly.express as px

from app_styles import SPAGHETTI_PLOT_AXIS_CONFIG


# Probably need a better name for these functions...


def get_dashboard_spaghetti(df_plot: pd.DataFrame,
                            spaghetti_color_map: dict):
    """
    Returns plotly.graph_objects.Figure
    """

    # Infuriating trying to get an empty lineplot NOT to display negative numbers
    #   -- still does this even with repeated attempts at overriding by adjusting
    #   "range"-related parameters :(
    # To fix... don't show ticks if plot is empty

    if df_plot.empty:
        is_nonempty_df = True
    else:
        is_nonempty_df = False

    fig = px.line(
        df_plot,
        x='day',
        y='result',
        color='simulation_idx',
        color_discrete_map=spaghetti_color_map)

    fig.update_traces(hovertemplate="Day %{x}<br>%{y:.1f} Infected<extra></extra>")
    fig.update_traces(line=dict(width=2))  # Reduce line thickness

    fig.update_layout(showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(
                          title="Day",
                          **SPAGHETTI_PLOT_AXIS_CONFIG,
                          linecolor="grey",
                          showticklabels=is_nonempty_df),
                      yaxis=dict(
                          title="Number of infected students",
                          **SPAGHETTI_PLOT_AXIS_CONFIG,
                          linecolor="black",
                          showticklabels=is_nonempty_df
                      ))

    return fig


def get_dashboard_results_fig(df_spaghetti: pd.DataFrame,
                              index_sim_closest_median: int,
                              nb_curves_displayed: int,
                              curve_selection_seed: int):

    # df_spaghetti must have columns "day", "result", "simulation_idx"

    light_grey = 'rgb(220, 220, 220)'

    color_map = {
        x: light_grey
        for x in df_spaghetti['simulation_idx']
    }
    color_map[index_sim_closest_median] = 'rgb(0, 153, 204)'  # blue

    possible_idx = df_spaghetti['simulation_idx']
    possible_idx = possible_idx[possible_idx != index_sim_closest_median]

    sample_idx = np.random.Generator(
        np.random.MT19937(curve_selection_seed)).choice(possible_idx,
                                                        nb_curves_displayed,
                                                        replace=False)

    sim_plot_df = pd.concat([
        df_spaghetti.loc[df_spaghetti['simulation_idx'].isin(sample_idx)],
        df_spaghetti.loc[df_spaghetti['simulation_idx'] == index_sim_closest_median]
    ])

    return get_dashboard_spaghetti(sim_plot_df, color_map)


# Lauren wanted the default plot to have the same axes as the generated plot
#   from simulations
EMPTY_SPAGHETTI_PLOT_INFECTED_MA = get_dashboard_spaghetti(
    pd.DataFrame(columns=['day', 'result', 'simulation_idx']), {})
