import measles_single_population as msp
import pandas as pd
import numpy as np
import plotly.express as px

from app_styles import SPAGHETTI_PLOT_AXIS_CONFIG


def get_spaghetti_plot_infected_ma(df_plot: pd.DataFrame,
                                   spaghetti_color_map: dict):
    """
    TODO: can be refactored to plot other dataframes too,
    not just infected moving average.

    Returns plotly.graph_objects.Figure
    """

    # Infuriating trying to get an empty lineplot NOT to display negative numbers
    #   -- still does this even with repeated attempts at overriding by adjusting
    #   "range"-related parameters :(
    # To fix... don't show ticks if plot is empty

    is_nonempty_df = not df_plot.empty

    fig = px.line(
        df_plot,
        x='day',
        y='number_infected_7_day_ma',
        color='simulation_idx',
        color_discrete_map=spaghetti_color_map,
        labels={'simulation_idx': '', 'number_infected': 'Number of students infected', 'day': 'Day DD',
                "number_infected_7_day_ma": "NN infected (7-day average)"},
    )

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


def create_data_spaghetti_plot_infected_ma(sim: msp.StochasticSimulations,
                                           nb_curves_displayed: int,
                                           curve_selection_seed: int):
    df_spaghetti_infected_ma = sim.df_spaghetti_infected_ma
    index_sim_closest_median = sim.index_sim_closest_median

    light_grey = 'rgb(220, 220, 220)'

    color_map = {
        x: light_grey
        for x in df_spaghetti_infected_ma['simulation_idx'].unique()
    }
    color_map[index_sim_closest_median] = 'rgb(0, 153, 204)'  # blue

    possible_idx = [
        x for x in df_spaghetti_infected_ma['simulation_idx'].unique()
        if x != index_sim_closest_median
    ]

    sample_idx = np.random.Generator(
        np.random.MT19937(curve_selection_seed)).choice(possible_idx,
                                                        nb_curves_displayed,
                                                        replace=False)

    sim_plot_df = pd.concat([
        df_spaghetti_infected_ma.loc[df_spaghetti_infected_ma['simulation_idx'].isin(sample_idx)],
        df_spaghetti_infected_ma.loc[df_spaghetti_infected_ma['simulation_idx'] == index_sim_closest_median]
    ])

    return sim_plot_df, color_map


# Lauren wanted the default plot to have the same axes as the generated plot
#   from simulations
EMPTY_SPAGHETTI_PLOT_INFECTED_MA = get_spaghetti_plot_infected_ma(
    pd.DataFrame(columns=['day', 'number_infected_7_day_ma', 'simulation_idx']), {})
