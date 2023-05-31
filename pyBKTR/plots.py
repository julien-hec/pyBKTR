import math
from itertools import cycle
from textwrap import wrap
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from pyBKTR.bktr import BKTRRegressor
from pyBKTR.utils import get_label_index_or_raise


def plot_temporal_betas(
    bktr_reg: BKTRRegressor,
    plot_feature_labels: list[str],
    spatial_point_label: str,
    show_figure: bool = True,
    fig_width: int = 850,
    fig_height: int = 550,
):
    """Create a plot of the beta values through time for a given spatial point and a set of
        feature labels.

    Args:
        bktr_reg (BKTRRegressor): BKTRRegressor object.
        plot_feature_labels (list[str]): List of feature labels to plot.
        spatial_point_label (str): Spatial point label to plot.
        show_figure (bool, optional): Whether to show the figure. Defaults to True.
        fig_width (int, optional): Figure width. Defaults to 850.
        fig_height (int, optional): Figure height. Defaults to 550.
    """
    if not bktr_reg.has_completed_sampling:
        raise RuntimeError('Plots can only be accessed after MCMC sampling.')
    # Verify all labels are valid
    get_label_index_or_raise(spatial_point_label, bktr_reg.spatial_labels, 'spatial')
    for feature_label in plot_feature_labels:
        get_label_index_or_raise(feature_label, bktr_reg.feature_labels, 'feature')

    scatters = []
    colorscale_hexas = px.colors.qualitative.Plotly
    color_cycle = cycle(colorscale_hexas)

    beta_summary_df: pd.DataFrame = bktr_reg.get_beta_summary_df(
        [spatial_point_label],
        None,
        plot_feature_labels,
    )
    for feature_label in plot_feature_labels:
        beta_df = beta_summary_df.loc[
            (spatial_point_label, slice(None), feature_label),
            ['Low2.5p', 'Mean', 'Up97.5p'],
        ]
        col_title = feature_label.replace('_', ' ').title()
        line_color = next(color_cycle)
        fill_rgba = _hex_to_rgba(line_color, 0.2)
        pctl_025 = beta_df['Low2.5p'].to_list()
        pctl_975 = beta_df['Up97.5p'].to_list()

        scatters.extend(
            [
                go.Scatter(
                    name=col_title,
                    x=bktr_reg.temporal_labels,
                    y=beta_df['Mean'],
                    mode='lines',
                    line=dict(color=line_color),
                ),
                go.Scatter(
                    name=f'{col_title} Bounds',
                    x=bktr_reg.temporal_labels + bktr_reg.temporal_labels[::-1],
                    y=pctl_975 + pctl_025[::-1],
                    mode='lines',
                    fillcolor=fill_rgba,
                    line=dict(width=0),
                    fill='toself',
                    hoverinfo='skip',
                    showlegend=False,
                ),
            ]
        )

    fig = go.Figure(scatters)
    fig.update_layout(
        yaxis_title='Value of coefficients',
        title=f'Location: {spatial_point_label.title()}',
        hovermode='x',
        width=fig_width,
        height=fig_height,
    )
    if show_figure:
        fig.show()
    return fig


def plot_spatial_betas(
    bktr_reg: BKTRRegressor,
    plot_feature_labels: list[str],
    temporal_point_label: str,
    nb_cols: int = 1,
    is_map: bool = True,
    mapbox_zoom: int = 9,
    use_dark_mode: bool = True,
    show_figure: bool = True,
    fig_width: int = 850,
    fig_height: int = 550,
):
    """Create a plot of beta values through space for a given temporal point and a set of
        feature labels.

    Args:
        bktr_reg (BKTRRegressor): BKTRRegressor object.
        plot_feature_labels (list[str]): List of feature labels to plot.
        temporal_point_label (str): Temporal point label to plot.
        nb_cols (int, optional): Number of columns in the plot. Defaults to 1.
        is_map (bool, optional): Whether to plot as a map. Defaults to True.
        mapbox_zoom (int, optional): Mapbox zoom. Defaults to 9. Only used if is_map is True.
        use_dark_mode (bool, optional): Whether to use dark mode. Defaults to True. Only used if
            is_map is True.
        show_figure (bool, optional): Whether to show the figure. Defaults to True.
        fig_width (int, optional): Figure width. Defaults to 850.
        fig_height (int, optional): Figure height. Defaults to 550.
    """
    if not bktr_reg.has_completed_sampling:
        raise RuntimeError('Plots can only be accessed after MCMC sampling.')
    # Verify all labels are valid
    get_label_index_or_raise(temporal_point_label, bktr_reg.temporal_labels, 'temporal')
    for feature_label in plot_feature_labels:
        get_label_index_or_raise(feature_label, bktr_reg.feature_labels, 'feature')

    beta_df = bktr_reg.beta_estimates.loc[(slice(None), temporal_point_label), plot_feature_labels]
    beta_df.melt(
        var_name='feature', value_vars=plot_feature_labels, ignore_index=False
    ).reset_index()

    feature_titles = [_get_feature_title(s) for s in plot_feature_labels]
    min_beta, max_beta = beta_df.min().min(), beta_df.max().max()
    nb_subplots = len(plot_feature_labels)
    nb_rows = math.ceil(nb_subplots / nb_cols)
    df_coord = bktr_reg.spatial_positions_df.copy()
    if df_coord.shape[1] != 2:
        raise ValueError('Spatial coordinates must have 2 columns to be plotted.')
    if not is_map:
        plot_df = beta_df.melt(
            var_name='feature', value_vars=plot_feature_labels, ignore_index=False
        ).reset_index()
        plot_df = plot_df.merge(df_coord.reset_index(), on='location')
        col_x, col_y = df_coord.columns
        fig = px.scatter(
            plot_df,
            facet_col='feature',
            x=col_x,
            y=col_y,
            color='value',
            height=fig_height,
            width=fig_width,
        )
        if show_figure:
            fig.show()
            return
        return fig
    fig = make_subplots(
        rows=nb_rows,
        cols=nb_cols,
        subplot_titles=feature_titles,
        specs=[[{'type': 'mapbox'} for _ in range(nb_cols)] for _ in range(nb_rows)],
    )
    df_coord.columns = ['lat', 'lon']
    lat_list = df_coord['lat'].to_list()
    lon_list = df_coord['lon'].to_list()
    for i, feature_label in enumerate(plot_feature_labels):
        beta_df_feature = beta_df.loc[(slice(None), temporal_point_label), feature_label]
        beta_col_list = beta_df_feature.to_list()
        fig.add_trace(
            go.Scattermapbox(
                lat=lat_list,
                lon=lon_list,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    cmin=min_beta,
                    cmax=max_beta,
                    color=beta_col_list,
                    size=7,
                    showscale=i == 0,
                    colorscale=px.colors.sequential.Viridis,
                    colorbar=dict(
                        title='Beta Values',
                        thickness=20,
                        titleside='top',
                        outlinecolor='rgba(68,68,68,0)',
                        ticks='outside',
                        ticklen=3,
                    ),
                ),
                text=[feature_titles],
            ),
            row=i // nb_cols + 1,
            col=i % nb_cols + 1,
        )
    fig.update_mapboxes(
        zoom=mapbox_zoom,
        style='carto-darkmatter' if use_dark_mode else 'carto-positron',
        center=go.layout.mapbox.Center(
            lat=sum(lat_list) / len(lat_list), lon=sum(lon_list) / len(lon_list)
        ),
    )
    fig.update_layout(
        showlegend=False,
        width=fig_width,
        height=fig_height,
    )
    if show_figure:
        fig.show()
        return
    return fig


def plot_beta_dists(
    bktr_reg: BKTRRegressor,
    labels_list: list[tuple[Any, Any, Any]],
    show_figure: bool = True,
    fig_width: int = 900,
    fig_height: int = 600,
):
    """Plot the distribution of beta values for a given list of labels.

    Args:
        bktr_reg (BKTRRegressor): BKTRRegressor object.
        labels_list (list[tuple[Any, Any, Any]]): List of labels (spatial, temporal, feature)
            for which to plot the beta distribution through iterations.
        show_figure (bool, optional): Whether to show the figure. Defaults to True.
        fig_width (int, optional): Figure width. Defaults to 900.
        fig_height (int, optional): Figure height. Defaults to 600.
    """
    if not bktr_reg.has_completed_sampling:
        raise RuntimeError('Plots can only be accessed after MCMC sampling.')
    betas_list = [bktr_reg.get_iterations_betas(*lab) for lab in labels_list]
    group_names = ['<br>'.join(lab) for lab in labels_list]
    fig = go.Figure()

    for i, betas in enumerate(betas_list):
        df = pd.DataFrame({'beta': betas, 'labels': [group_names[i]] * len(betas)})
        fig.add_trace(
            go.Violin(
                x=df['labels'],
                y=df['beta'],
                name=group_names[i],
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        showlegend=False,
        width=fig_width,
        height=fig_height,
        xaxis={'type': 'category'},
        yaxis_title='Beta Value',
        title=(
            'Posterior distribution of beta values per given'
            ' spatial point, temporal point and feature'
        ),
    )
    if show_figure:
        fig.show()
        return
    return fig


def plot_covariates_beta_dists(
    bktr_reg: BKTRRegressor,
    feature_labels: list[Any] | None = None,
    show_figure: bool = True,
    fig_width: int = 900,
    fig_height: int = 600,
):
    """Plot the distribution of beta estimates regrouped by covariates.

    Args:
        bktr_reg (BKTRRegressor): BKTRRegressor object.
        feature_labels (list[Any] | None, optional): List of feature labels labels for
            which to plot the beta estimates distribution. If None plot for all features.
        show_figure (bool, optional): Whether to show the figure. Defaults to True.
        fig_width (int, optional): Figure width. Defaults to 900.
        fig_height (int, optional): Figure height. Defaults to 600.
    """
    if not bktr_reg.has_completed_sampling:
        raise RuntimeError('Plots can only be accessed after MCMC sampling.')
    if feature_labels is None:
        feature_labels = bktr_reg.feature_labels
    fig = go.Figure()
    for lab in feature_labels:
        if lab not in bktr_reg.beta_estimates.columns:
            raise ValueError(f'Invalid covariate label: {lab}')
        df = pd.DataFrame({'beta': bktr_reg.beta_estimates[[lab]][lab].to_list(), 'feature': lab})
        fig.add_trace(
            go.Violin(
                x=df['feature'],
                y=df['beta'],
                name=lab,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        showlegend=False,
        width=fig_width,
        height=fig_height,
        xaxis={'type': 'category'},
        yaxis_title='Beta Value',
        title='Distribution of beta estimates by feature across time and space',
    )
    if show_figure:
        fig.show()
        return
    return fig


def plot_hyperparams_dists(
    bktr_reg: BKTRRegressor,
    hyperparameters: list[str] | None = None,
    show_figure: bool = True,
    fig_width: int = 900,
    fig_height: int = 600,
):
    """Plot the distribution of hyperparameters through iterations.

    Args:
        hyperparameters (list[str] | None, optional): List of hyperparameters to plot.
            If None, plot all hyperparameters. Defaults to None.
        show_figure (bool, optional): Whether to show the figure. Defaults to True.
        fig_width (int, optional): Figure width. Defaults to 900.
        fig_height (int, optional): Figure height. Defaults to 600.
    """
    if not bktr_reg.has_completed_sampling:
        raise RuntimeError('Plots can only be accessed after MCMC sampling.')
    hparams_per_iter_df = bktr_reg.result_logger.hyperparameters_per_iter_df
    hparams = hparams_per_iter_df.columns if hyperparameters is None else hyperparameters
    fig = go.Figure()
    hparam_diff = set(hparams) - set(hparams_per_iter_df.columns)
    if hparam_diff:
        formatted_available_params = '\n'.join(hparams_per_iter_df.columns)
        formatted_hparam_diff = ', '.join(hparam_diff)
        raise ValueError(
            f'Hyperparameter(s) {formatted_hparam_diff} not found.'
            f' Available hyperparameters are:\n{formatted_available_params}'
        )
    for hparam in hparams:
        df = hparams_per_iter_df[[hparam]]
        fig.add_trace(
            go.Violin(
                x=[hparam] * len(df),
                y=df[hparam],
                name=hparam,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        showlegend=False,
        width=fig_width,
        height=fig_height,
        xaxis={'type': 'category'},
        yaxis_title='Hyperparameter Value',
        title='Posterior distribution of BKTR hyperparameters',
    )
    if show_figure:
        fig.show()
        return
    return fig


def plot_hyperparams_traceplot(
    bktr_reg: BKTRRegressor,
    hyperparameters: list[str] | None = None,
    show_figure: bool = True,
    fig_width: int = 800,
    fig_height: int = 550,
):
    """Plot the distribution of hyperparameters through iterations.

    Args:
        hyperparameters (list[str] | None, optional): List of hyperparameters to plot.
            If None, plot all hyperparameters. Defaults to None.
        show_figure (bool, optional): Whether to show the figure. Defaults to True.
        fig_width (int, optional): Figure width. Defaults to 800.
        fig_height (int, optional): Figure height. Defaults to 550.
    """
    if not bktr_reg.has_completed_sampling:
        raise RuntimeError('Plots can only be accessed after MCMC sampling.')
    hparams_per_iter_df = bktr_reg.result_logger.hyperparameters_per_iter_df
    hparams = hparams_per_iter_df.columns if hyperparameters is None else hyperparameters
    hparam_diff = set(hparams) - set(hparams_per_iter_df.columns)
    if hparam_diff:
        formatted_available_params = '\n'.join(hparams_per_iter_df.columns)
        formatted_hparam_diff = ', '.join(hparam_diff)
        raise ValueError(
            f'Hyperparameter(s) {formatted_hparam_diff} not found.'
            f' Available hyperparameters are:\n{formatted_available_params}'
        )
    df = hparams_per_iter_df[hparams].copy()
    df.reset_index(inplace=True)
    df = df.melt(id_vars=['Sampling Iter'], var_name='Hyperparameter', value_name='Value')

    fig = px.line(df, x='Sampling Iter', y='Value', color='Hyperparameter')
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        xaxis={'type': 'category'},
        yaxis_title='Hyperparameter Value',
        title='Hyperparameter values through sampling iterations (Traceplot)',
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.3,
            'x': 1,
            'xanchor': 'right',
        },
    )
    if show_figure:
        fig.show()
        return
    return fig


def _get_feature_title(feature_name: str) -> str:
    feature_title = feature_name.replace('_', ' ').title()
    return f'{feature_title} Beta Values'


def _hex_to_rgba(hex: str, transparency: float) -> str:
    col_hex = hex.lstrip('#')
    # Transform each hex in dec
    col_rgb = tuple(int(s, 16) for s in wrap(col_hex, 2))
    return f'rgba({col_rgb[0]}, {col_rgb[1]}, {col_rgb[2]}, {transparency})'
