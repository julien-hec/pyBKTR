import math
from itertools import cycle
from textwrap import wrap

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from pyBKTR.utils import get_label_index_or_raise


class BKTRBetaPlotMaker:
    def __init__(self, beta_summary_df, spatial_labels, temporal_labels, feature_labels) -> None:
        self.beta_summary_df = beta_summary_df
        self.spatial_labels = spatial_labels
        self.temporal_labels = temporal_labels
        self.feature_labels = feature_labels

    def plot_temporal_betas(
        self,
        plot_feature_labels: list[str],
        spatial_point_label: str,
        show_figure: bool = True,
        fig_width: int = 850,
        fig_height: int = 550,
    ):
        # Verify all labels are valid
        get_label_index_or_raise(spatial_point_label, self.spatial_labels, 'spatial')
        for feature_label in plot_feature_labels:
            get_label_index_or_raise(feature_label, self.feature_labels, 'feature')

        scatters = []
        colorscale_hexas = px.colors.qualitative.Plotly
        color_cycle = cycle(colorscale_hexas)

        for feature_label in plot_feature_labels:
            beta_df = self.beta_summary_df.loc[
                (spatial_point_label, slice(None), feature_label),
                ['2.5th Percentile', 'Mean', '97.5th Percentile'],
            ]
            col_title = feature_label.replace('_', ' ').title()
            line_color = next(color_cycle)
            fill_rgba = self.hex_to_rgba(line_color, 0.2)
            pctl_025 = beta_df['2.5th Percentile'].to_list()
            pctl_975 = beta_df['97.5th Percentile'].to_list()

            scatters.extend(
                [
                    go.Scatter(
                        name=col_title,
                        x=self.temporal_labels,
                        y=beta_df['Mean'],
                        mode='lines',
                        line=dict(color=line_color),
                    ),
                    go.Scatter(
                        name=f'{col_title} Bounds',
                        x=self.temporal_labels + self.temporal_labels[::-1],
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
        self,
        plot_feature_labels: list[str],
        temporal_point_label: str,
        geo_coordinates: pd.DataFrame,
        nb_cols: int = 1,
        mapbox_zoom: int = 9,
        use_dark_mode: bool = True,
        show_figure: bool = True,
        fig_width: int = 850,
        fig_height: int = 550,
    ):
        # Verify all labels are valid
        get_label_index_or_raise(temporal_point_label, self.temporal_labels, 'temporal')
        for feature_label in plot_feature_labels:
            get_label_index_or_raise(feature_label, self.feature_labels, 'feature')

        beta_df = self.beta_summary_df.loc[
            (slice(None), temporal_point_label, plot_feature_labels), ['Mean']
        ]

        feature_titles = [self.get_feature_title(s) for s in plot_feature_labels]
        min_beta, max_beta = beta_df['Mean'].min(), beta_df['Mean'].max()
        nb_subplots = len(plot_feature_labels)
        nb_rows = math.ceil(nb_subplots / nb_cols)
        fig = make_subplots(
            rows=nb_rows,
            cols=nb_cols,
            subplot_titles=feature_titles,
            specs=[[{'type': 'mapbox'} for _ in range(nb_cols)] for _ in range(nb_rows)],
        )
        df_coord = geo_coordinates.copy()
        df_coord.columns = ['lat', 'lon']
        lat_list = df_coord['lat'].to_list()
        lon_list = df_coord['lon'].to_list()
        for i, feature_label in enumerate(plot_feature_labels):
            beta_df_feature = beta_df.loc[
                (slice(None), temporal_point_label, feature_label), ['Mean']
            ]
            beta_col_list = beta_df_feature['Mean'].to_list()
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

    @staticmethod
    def get_feature_title(feature_name: str) -> str:
        feature_title = feature_name.replace('_', ' ').title()
        return f'{feature_title} Beta Values'

    @staticmethod
    def hex_to_rgba(hex: str, transparency: float) -> str:
        col_hex = hex.lstrip('#')
        # Transform each hex in dec
        col_rgb = tuple(int(s, 16) for s in wrap(col_hex, 2))
        return f'rgba({col_rgb[0]}, {col_rgb[1]}, {col_rgb[2]}, {transparency})'
