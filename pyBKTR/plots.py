import math
from itertools import cycle
from textwrap import wrap

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from pyBKTR.bktr import BKTRRegressor


class BKTRBetaPlotMaker:
    def __init__(
        self,
        bktr_regressor: BKTRRegressor,
        spatial_feature_labels: list[str],
        temporal_feature_labels: list[str],
        spatial_points_labels: list[str],
        temporal_points_labels: list[str],
    ) -> None:
        self.bktr_regressor = bktr_regressor
        self.spatial_feature_labels = spatial_feature_labels
        self.temporal_feature_labels = temporal_feature_labels
        self.spatial_points_labels = spatial_points_labels
        self.temporal_points_labels = temporal_points_labels
        self.feature_labels = ['_INTERSECT_', *spatial_feature_labels, *temporal_feature_labels]

    def get_beta_est_stdev_dfs(
        self,
        plot_feature_labels: list[str],
        plot_point_label: str,
        is_temporal_plot: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Get label indexes
        feature_labels_indexes = [self.feature_labels.index(s) for s in plot_feature_labels]

        # Get beta estimates and standard deviations
        beta_est = self.bktr_regressor.beta_estimates
        beta_stdev = self.bktr_regressor.beta_stdev
        if is_temporal_plot:
            point_label_index = self.spatial_points_labels.index(plot_point_label)
            beta_est_values = beta_est[point_label_index, :, feature_labels_indexes]
            beta_stdev_values = beta_stdev[point_label_index, :, feature_labels_indexes]
            beta_est_df = pd.DataFrame(
                beta_est_values, columns=plot_feature_labels, index=self.temporal_points_labels
            )
            beta_stdev_df = pd.DataFrame(
                beta_stdev_values, columns=plot_feature_labels, index=self.temporal_points_labels
            )
            return beta_est_df, beta_stdev_df

        # Get only spatial estimates for spatial plots
        point_label_index = self.temporal_points_labels.index(plot_point_label)
        beta_est_values = beta_est[:, point_label_index, feature_labels_indexes]
        beta_stdev_values = beta_stdev[:, point_label_index, feature_labels_indexes]
        beta_est_df = pd.DataFrame(
            beta_est_values, columns=plot_feature_labels, index=self.spatial_points_labels
        )
        return beta_est_df, None

    def create_temporal_beta_plot(
        self,
        plot_feature_labels: list[str],
        plot_point_label: str,
        show_figure: bool = True,
        colorscale_hexas=px.colors.qualitative.Plotly,
    ):
        beta_est_df, beta_stdev_df = self.get_beta_est_stdev_dfs(
            plot_feature_labels, plot_point_label, is_temporal_plot=True
        )
        scatters = []
        color_cycle = cycle(colorscale_hexas)

        for col_name in plot_feature_labels:
            col_tile = col_name.replace('_', ' ').title()
            df_col_est = beta_est_df[col_name]
            df_col_stdev = beta_stdev_df[col_name]

            line_color = next(color_cycle)
            fill_rgba = self.hex_rgba(line_color, 0.2)

            x_index = beta_est_df.index.to_list()
            upper_stdev = (df_col_est + df_col_stdev).to_list()
            lower_stdev = (df_col_est - df_col_stdev).to_list()

            scatters.extend(
                [
                    go.Scatter(
                        name=col_tile,
                        x=x_index,
                        y=df_col_est,
                        mode='lines',
                        line=dict(color=line_color),
                    ),
                    go.Scatter(
                        name=f'{col_tile} Bounds',
                        x=x_index + x_index[::-1],
                        y=upper_stdev + lower_stdev[::-1],
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
            title=f'Location: {plot_point_label.title()}',
            hovermode='x',
            width=850,
            height=550,
        )
        if show_figure:
            fig.show()
        return fig

    def create_spatial_beta_plots(
        self,
        plot_feature_labels: list[str],
        plot_point_label: str,
        geo_coordinates: list[list[float, float]],
        nb_cols: int = 1,
        mapbox_zoom: int = 9,
        use_dark_mode: bool = True,
    ):
        beta_est_df, _ = self.get_beta_est_stdev_dfs(
            plot_feature_labels, plot_point_label, is_temporal_plot=False
        )
        feature_titles = [self.get_feature_title(s) for s in plot_feature_labels]
        min_beta, max_beta = beta_est_df.min().min(), beta_est_df.max().max()
        nb_subplots = len(plot_feature_labels)
        nb_rows = math.ceil(nb_subplots / nb_cols)
        fig = make_subplots(
            rows=nb_rows,
            cols=nb_cols,
            subplot_titles=feature_titles,
            specs=[[{'type': 'mapbox'} for _ in range(nb_cols)] for _ in range(nb_rows)],
        )
        df_coord = pd.DataFrame(
            geo_coordinates, columns=['lat', 'lon'], index=self.spatial_points_labels
        )
        lat_list = df_coord['lat'].to_list()
        lon_list = df_coord['lon'].to_list()
        for i, feature_label in enumerate(plot_feature_labels):
            beta_col_list = beta_est_df[feature_label].to_list()
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
        fig.update_layout(showlegend=False)
        fig.show()

    @staticmethod
    def get_feature_title(feature_name: str) -> str:
        feature_title = feature_name.replace('_', ' ').title()
        return f'{feature_title} Beta Values'

    @staticmethod
    def hex_rgba(hex: str, transparency: float) -> str:
        col_hex = hex.lstrip('#')
        # Transform each hex in dec
        col_rgb = tuple(int(s, 16) for s in wrap(col_hex, 2))
        return f'rgba({col_rgb[0]}, {col_rgb[1]}, {col_rgb[2]}, {transparency})'
