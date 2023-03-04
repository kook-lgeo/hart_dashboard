from dash import Dash, dcc, dash_table, html, Input, Output, ctx, callback
import dash
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
from dash.dash_table.Format import Format, Scheme, Sign, Symbol, Group
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import warnings
import json
import geopandas as gpd
import fiona
# from app import app
# from pages import main

fiona.supported_drivers  


warnings.filterwarnings("ignore")

# Importing income data

engine = create_engine('sqlite:///sources//hart.db')


# Importing Geo Code Information

mapped_geo_code = pd.read_sql_table('geocodes_integrated', engine.connect())
df_geo_list = pd.read_sql_table('geocodes', engine.connect())
df_region_list = pd.read_sql_table('regioncodes', engine.connect())
df_region_list.columns = df_geo_list.columns
df_province_list = pd.read_sql_table('provincecodes', engine.connect())
df_province_list.columns = df_geo_list.columns

# Importing Projection Data

df_csd_proj = pd.read_sql_table('csd_hh_projections', engine.connect())
df_cd_proj = pd.read_sql_table('cd_hh_projections', engine.connect())
df_cd_grow = pd.read_sql_table('cd_growthrates', engine.connect())

# Merging Projection data with Geography codes

df_csd_proj_merged = df_geo_list.merge(df_csd_proj, how = 'left', on = 'Geo_Code')
df_cd_proj_merged = df_region_list.merge(df_cd_proj, how = 'left', on = 'Geo_Code')
df_cd_grow_merged = df_region_list.merge(df_cd_grow, how = 'left', on = 'Geo_Code')

# New projection data
updated_csd = pd.read_csv('./sources/updated_csd.csv')
updated_cd = pd.read_csv('./sources/updated_cd.csv')

# Configuration for plot icons

config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['zoom', 'lasso2d', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']}

# Preprocessing
fig = px.line(x = ['Not Available in CD/Province level. Please select CSD level region'], y = ['Not Available in CD/Province level. Please select CSD level region'])

table = pd.DataFrame({'Not Available in CD/Province level. Please select CSD level region':[0]})

colors = ['#D7F3FD', '#88D9FA', '#39C0F7', '#099DD7', '#044762']
hh_colors = ['#D8EBD4', '#93CD8A', '#3DB54A', '#297A32', '#143D19']
hh_type_color = ['#3949CE', '#3EB549', '#39C0F7']
columns_color_fill = ['#F3F4F5', '#EBF9FE', '#F0FAF1']
map_colors_wo_black = ['#39C0F7', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#F4F739']
map_colors_w_black = ['#000000', '#39C0F7', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#F4F739']
modebar_color = '#099DD7'
modebar_activecolor = '#044762'

comparison_font_size = '0.7em'

# Setting layout for dashboard

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
# server = app.server

# dash.register_page(__name__)

layout = html.Div(children = [
        dcc.Store(id='area-scale-store', storage_type='local'),
        dcc.Store(id='main-area', storage_type='local'),
        dcc.Store(id='comparison-area', storage_type='local'),

        html.Div(
        children = [
            html.Div([
                html.H2(children = html.Strong("2026 Household Projections by Household Size and Income Category"), id = 'home')
            ], className = 'title-lgeo'),

        # 2026 Projections by HH Size and Income Level


            # Table

            html.Div([

                html.Div([
                   
                    html.H3(children = html.Strong('2026 Household Projections by Income Category'), className = 'subtitle-lgeo'),

                    html.Div(children = [ 

                        dash_table.DataTable(
                            id='datatable5-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                            ],
                            data=table.to_dict('records'),
                            editable=True,
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable=False,
                            row_selectable=False,
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 10,
                            style_cell = {'font-family': 'Bahnschrift'},
                            merge_duplicate_headers=True,
                            export_format="csv",
                            # style_table={'minWidth': '100%'},
                            style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                        ),
                    ], className = 'pg3-table-lgeo'
                    ),

                    html.Div(id='datatable5-interactivity-container'),


                    # Graphs

                    html.Div(children = [ 

                        dcc.Graph(
                            id='graph9',
                            figure=fig,
                            config = config,
                        ),

                    ], className = 'pg3-plot-lgeo'
                    ),

                ], className = 'pg3-table-plot-box-lgeo'),



                html.Div([
                   
                    html.H3(children = html.Strong('2026 Household Projections by Household Size'), className = 'subtitle-lgeo'),


                    html.Div([
                        dash_table.DataTable(
                            id='datatable6-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                            ],
                            data=table.to_dict('records'),
                            editable=True,
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable=False,
                            row_selectable=False,
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 10,
                            style_cell = {'font-family': 'Bahnschrift'},
                            merge_duplicate_headers=True,
                            export_format="csv",
                            # style_table={'minWidth': '100%'},
                            style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                        ),
                    ], className = 'pg3-table-lgeo'
                    ),

                    html.Div(id='datatable6-interactivity-container'),


                    # Graphs

                    html.Div(children = [ 
                        dcc.Graph(
                            id='graph10',
                            figure=fig,
                            config = config,
                        ),                            
                    ], className = 'pg3-plot-lgeo'
                    ),
                ], className = 'pg3-table-plot-box-lgeo'),




                html.Div([
                   
                    html.H3(children = html.Strong('2026 Projected Households'), className = 'subtitle-lgeo'),

                    html.Div([
                        dash_table.DataTable(
                            id='datatable-h-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                            ],
                            data=table.to_dict('records'),
                            editable=True,
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable=False,
                            row_selectable=False,
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 10,
                            style_cell = {'font-family': 'Bahnschrift'},
                            merge_duplicate_headers=True,
                            export_format="csv",
                            # style_table={'minWidth': '100%'},
                            style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                        ),
                    ], className = 'pg3-table-lgeo'
                    ),

                    html.Div(id='datatable-h-interactivity-container'),

                    # Graphs

                    html.Div(children = [ 

                        dcc.Graph(
                            id='graph-h',
                            figure=fig,
                            config = config,
                        )

                    ], className = 'pg3-plot-lgeo'
                    ),
                ], className = 'pg3-table-plot-box-lgeo'),



                html.Div([
                   
                    html.H3(children = html.Strong('2026 Projected Household Change (Delta)'), className = 'table-title'),

                    html.Div([
                        dash_table.DataTable(
                            id='datatable7-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                            ],
                            data=table.to_dict('records'),
                            editable=True,
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable=False,
                            row_selectable=False,
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 10,
                            style_cell = {'font-family': 'Bahnschrift'},
                            merge_duplicate_headers=True,
                            export_format="csv",
                            # style_table={'minWidth': '100%'},
                            style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                        ),
                    ], className = 'pg3-table-lgeo'
                    ),


                    html.Div(id='datatable7-interactivity-container'),


                    # Graphs

                    html.Div(children = [ 

                        dcc.Graph(
                            id='graph11',
                            figure=fig,
                            config = config,
                        )
                    ], className = 'pg3-plot-lgeo'
                    ),
                ], className = 'pg3-table-plot-box-lgeo'),

                html.Div([
                    html.H3(children = html.Strong('Municipal vs Regional Rate'), className = 'subtitle-lgeo'),
                    html.Strong('Comparing a local community’s growth rates to the growth rate of the region allows for insight into if the community is matching regional patterns of change. A significantly lower growth rate in a community compared to its region could be caused by artificially suppressed growth due to low housing supply. A significantly higher growth rate in a community compared to its region could be caused by spillover in interconnected urban communities. The following graphs and tables illustrate 2026 household growth using both the community and regional growth rates.')
                ], className = 'muni-reg-text-lgeo'),

                html.Div([
                   
                    html.H3(children = html.Strong('2026 HH - Municipal and Regional Growth Rates by HH Income'), className = 'subtitle-lgeo'),

                    html.Div([

                        dash_table.DataTable(
                            id='datatable8-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                            ],
                            data=table.to_dict('records'),
                            editable=True,
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable=False,
                            row_selectable=False,
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 10,
                            style_cell = {'font-family': 'Bahnschrift'},
                            merge_duplicate_headers=True,
                            export_format="csv",
                            # style_table={'minWidth': '100%'},
                            style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                        ),
                    ], className = 'pg3-table-lgeo'
                    ),

                    html.Div(id='datatable8-interactivity-container'),


                    # Graphs

                    html.Div(children = [ 
                        dcc.Graph(
                            id='graph12',
                            figure=fig,
                            config = config,
                        )
                    ],
                    className = 'pg3-plot-lgeo'
                    ),
                ], className = 'pg3-table-plot-box-lgeo'),


                html.Div([
                   
                    html.H3(children = html.Strong('2026 HH - Municipal and Regional Growth Rates by HH Size'), className = 'subtitle-lgeo'),

                    html.Div([
                        dash_table.DataTable(
                            id='datatable9-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                            ],
                            data=table.to_dict('records'),
                            editable=True,
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable=False,
                            row_selectable=False,
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 10,
                            style_cell = {'font-family': 'Bahnschrift'},
                            merge_duplicate_headers=True,
                            export_format="csv",
                            # style_table={'minWidth': '100%'},
                            style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                        ),

                    ], className = 'pg3-table-lgeo'
                    ),

                    html.Div(id='datatable9-interactivity-container'),

                    # Graphs

                    html.Div(children = [ 

                        dcc.Graph(
                            id='graph13',
                            figure=fig,
                            config = config,
                            )
                    ],
                    className = 'pg3-plot-lgeo'
                    ),
                ], className = 'pg3-table-plot-box-lgeo'),


        ]),


        ], className = 'dashboard-lgeo'
    ), 
], className = 'background-lgeo'#style = {'backgroud-color': '#fffced'}
)



# new plot 1 for projection

def plot1_new_projection(geo, IsComparison):

    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Geo_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")

    updated_csd_filtered_2016_plot1 = updated_csd_filtered[[
        'Total - Private households by household type including census family structure -   Households with income 20% or under of area median household income (AMHI) - Total - Household size', 
        'Total - Private households by household type including census family structure -   Households with income 21% to 50% of AMHI - Total - Household size',
        'Total - Private households by household type including census family structure -   Households with income 51% to 80% of AMHI - Total - Household size',
        'Total - Private households by household type including census family structure -   Households with income 81% to 120% of AMHI - Total - Household size',
        'Total - Private households by household type including census family structure -   Households with income 121% or over of AMHI - Total - Household size'
        ]].T.reset_index().drop(columns = ['index'])

    updated_csd_filtered_2026_plot1 = updated_csd_filtered[[
        '2026 Population Delta with income 20% or under of area median household income (AMHI)',
        '2026 Population Delta with income 21% to 50% of AMHI',
        '2026 Population Delta with income 51% to 80% of AMHI',
        '2026 Population Delta with income 81% to 120% of AMHI',
        '2026 Population Delta with income 121% or over of AMHI'
        ]].T.reset_index().drop(columns = ['index'])

    income_category = ['Very Low Income', 'Low Income', 'Moderate Income', 'Median Income', 'High Income']


    table1 = pd.DataFrame({'Income Category': income_category, 
                            'Category': (['2016 Pop'] * len(income_category)),
                            'Pop': updated_csd_filtered_2016_plot1.iloc[:,0]})
    
    table1_2016 = table1.copy()

    table1['2026 Delta'] = np.round(updated_csd_filtered_2026_plot1.iloc[:,0],0)
    table1 = table1.drop(columns = ['Category'])

    if IsComparison != True:
        table1.columns = ['HH Income Category', '2016 HHs', '2026 Change in HH']
    else:
        table1.columns = ['HH Income Category', '2016 HHs ', '2026 Change in HH ']

    plot_df = pd.concat([table1_2016,
                        pd.DataFrame({'Income Category': income_category,
                            'Category': (['2026 Delta'] * len(income_category)),
                            'Pop': np.round(updated_csd_filtered_2026_plot1.iloc[:,0],0)})])
    
    return plot_df, table1
    


@callback(
    Output('datatable5-interactivity', 'columns'),
    Output('datatable5-interactivity', 'data'),
    Output('datatable5-interactivity', 'style_data_conditional'),
    Output('datatable5-interactivity', 'style_cell_conditional'),
    Output('graph9', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('area-scale-store', 'data'),
    Input('datatable5-interactivity', 'selected_columns'),
)
def update_geo_figure6(geo, geo_c, scale, selected_columns):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == scale:
            geo = geo
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        plot_df, table1 = plot1_new_projection(geo, False)

        fig_new_proj_1 = go.Figure()

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['Income Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ))

        fig_new_proj_1.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'2026 Household Projections -<br>{geo}', legend_title = "Category")
        fig_new_proj_1.update_xaxes(fixedrange = True)
        fig_new_proj_1.update_yaxes(fixedrange = True)

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1


    else:

        if "to-geography-1" == scale:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        plot_df, table1 = plot1_new_projection(geo, False)

        fig_new_proj_1 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['Income Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 1)

        col_list = []

        for i in table1.columns:
            if i == 'Income Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        plot_df_c, table1_c = plot1_new_projection(geo_c, True)

        for i, c in zip(plot_df_c['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df_c.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['Income Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                showlegend = False, 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'Income Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_new_proj_1.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'2026 Household Projections', legend_title = "Category")
        fig_new_proj_1.update_yaxes(range=[0, max(plot_df.groupby('Income Category')['Pop'].sum().max(), plot_df_c.groupby('Income Category')['Pop'].sum().max())+100])
        fig_new_proj_1.update_xaxes(fixedrange = True)
        fig_new_proj_1.update_yaxes(fixedrange = True)

        table1_j = table1.merge(table1_c, how = 'left', on = 'HH Income Category')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1

# new plot 2 for projection

def plot2_new_projection(geo, IsComparison):

    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Geo_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")

    updated_csd_filtered_2016_plot2 = updated_csd_filtered[[
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   1pp',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   2pp',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   3pp',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   4pp',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   5pp'
        ]].T.reset_index().drop(columns = ['index'])

    updated_csd_filtered_2026_plot2 = updated_csd_filtered[[
    '2026 Population Delta 1pp HH',
    '2026 Population Delta 2pp HH',
    '2026 Population Delta 3pp HH',
    '2026 Population Delta 4pp HH',
    '2026 Population Delta 5pp HH'
        ]].T.reset_index().drop(columns = ['index'])

    hh_category = ['1 Person', '2 People', '3 People', '4 People', '5+ People']


    table2 = pd.DataFrame({'HH Category': hh_category, 
                            'Category': (['2016 Pop'] * len(hh_category)),
                            'Pop': updated_csd_filtered_2016_plot2.iloc[:,0]})
    
    table2_2016 = table2.copy()

    table2['2026 Delta'] = np.round(updated_csd_filtered_2026_plot2.iloc[:,0],0)
    table2 = table2.drop(columns = ['Category'])

    if IsComparison != True:
        table2.columns = ['HH Size', '2016 HHs', '2026 Change in HH']
    else:
        table2.columns = ['HH Size', '2016 HHs ', '2026 Change in HH ']

    plot_df = pd.concat([table2_2016,
                        pd.DataFrame({'HH Category': hh_category,
                            'Category': (['2026 Delta'] * len(hh_category)),
                            'Pop': np.round(updated_csd_filtered_2026_plot2.iloc[:,0],0)})])
    
    return plot_df, table2

@callback(
    Output('datatable6-interactivity', 'columns'),
    Output('datatable6-interactivity', 'data'),
    Output('datatable6-interactivity', 'style_data_conditional'),
    Output('datatable6-interactivity', 'style_cell_conditional'),
    Output('graph10', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('area-scale-store', 'data'),
    Input('datatable6-interactivity', 'selected_columns'),
)
def update_geo_figure7(geo, geo_c, scale, selected_columns):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == scale:
            geo = geo
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        plot_df, table1 = plot2_new_projection(geo, False)

        fig_new_proj_1 = go.Figure()

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ))

        fig_new_proj_1.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'2026 Household Projections -<br>{geo}', legend_title = "Category")
        fig_new_proj_1.update_xaxes(fixedrange = True)
        fig_new_proj_1.update_yaxes(fixedrange = True)

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1


    else:

        if "to-geography-1" == scale:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        plot_df, table1 = plot2_new_projection(geo, False)

        fig_new_proj_1 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 1)

        col_list = []

        for i in table1.columns:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        plot_df_c, table1_c = plot2_new_projection(geo_c, True)

        for i, c in zip(plot_df_c['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df_c.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_new_proj_1.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', showlegend = False, plot_bgcolor='#F8F9F9', title = f'2026 Household Projections', legend_title = "Category")
        fig_new_proj_1.update_yaxes(range=[0, max(plot_df.groupby('HH Category')['Pop'].sum().max(), plot_df_c.groupby('HH Category')['Pop'].sum().max())+100])
        fig_new_proj_1.update_xaxes(fixedrange = True)
        fig_new_proj_1.update_yaxes(fixedrange = True)

        table1_j = table1.merge(table1_c, how = 'left', on = 'HH Size')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1




# new plot h for projection

income_col_list = ['20% or under of area median household income (AMHI)', 
                    '21% to 50% of AMHI', 
                    '51% to 80% of AMHI', 
                    '81% to 120% of AMHI', 
                    '121% or over of AMHI']

pp_list = ['1pp', '2pp', '3pp', '4pp', '5pp']


def projections_2026_hh_size(geo, IsComparison):
    
    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, 'Geo_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")


    income_l = []
    pp_l = []
    result_csd_l = []


    for i in income_col_list:
        for p in pp_list:
            col_format = f'2026 Projected {p} HH with income {i}'
            income_l.append(i)
            pp_l.append(p)
            result_csd_l.append(updated_csd_filtered[col_format].tolist()[0])

    income_l = ['Very Low Income'] * 5 + ['Low Income'] * 5 + ['Moderate Income'] * 5 + ['Median Income'] * 5 + ['High Income'] * 5

    table3 = pd.DataFrame({'Income Category': income_l, 'HH Category': pp_l, 'CSD_Projection': np.round(result_csd_l,0)})
    table3 = table3.fillna(0)
    # table3_csd = table3.pivot_table(values='CSD_Projection', index=['Income Category'], columns=['HH Category'], sort = False)
    table3_csd = table3.pivot_table(values='CSD_Projection', index=['HH Category'], columns=['Income Category'], sort = False)
    table3_csd = table3_csd.reset_index()
    table3_csd = table3_csd.replace([np.inf, -np.inf], 0)
    table3_csd['HH Category'] = ['1 person', '2 people', '3 people', '4 people', '5+ people']

    table3_csd_plot = table3_csd.replace([np.inf, -np.inf], 0)
#     table3_csd_plot = pd.melt(table3_csd_plot, id_vars = 'Income Category', value_vars = ['1pp', '2pp', '3pp', '4pp', '5pp'])
    table3_csd_plot = pd.melt(table3_csd_plot, id_vars = 'HH Category', value_vars = ['Very Low Income', 'Low Income', 'Moderate Income',
           'Median Income', 'High Income'])
    table3_csd = table3_csd.replace([np.inf, -np.inf], 0)
    row_total_csd = table3_csd.sum(axis=0)
    row_total_csd[0] = 'Total'
    table3_csd.loc[5, :] = row_total_csd

    if IsComparison != True:
        table3_csd.columns = ['HH Income Category', 'Very Low Income', 'Low Income', 'Moderate Income',
       'Median Income', 'High Income']
        table3_csd['Total'] = table3_csd.sum(axis=1)

    else:
        table3_csd.columns = ['HH Income Category', 'Very Low Income ', 'Low Income ', 'Moderate Income ',
       'Median Income ', 'High Income ']
        table3_csd['Total '] = table3_csd.sum(axis=1)
        # table3_csd = table3_csd.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })

    return table3_csd, table3_csd_plot

@callback(
    Output('datatable-h-interactivity', 'columns'),
    Output('datatable-h-interactivity', 'data'),
    Output('datatable-h-interactivity', 'style_data_conditional'),
    Output('datatable-h-interactivity', 'style_cell_conditional'),
    Output('graph-h', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('area-scale-store', 'data'),
    Input('datatable-h-interactivity', 'selected_columns'),
)
def update_geo_figure_h(geo, geo_c, scale, selected_columns):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == scale:
            geo = geo
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        table1, table1_csd_plot = projections_2026_hh_size(geo, False)

        fig_csd = go.Figure()
        for i, c in zip(table1_csd_plot['Income Category'].unique(), colors):
            plot_df_frag = table1_csd_plot.loc[table1_csd_plot['Income Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ))

        fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='relative', plot_bgcolor='#F8F9F9', title = f'2026 Projected Households -<br>{geo}', legend_title = "Income Category")
        fig_csd.update_xaxes(fixedrange = True)
        fig_csd.update_yaxes(fixedrange = True)

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_csd


    else:

        if "to-geography-1" == scale:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        table1, table1_csd_plot = projections_2026_hh_size(geo, False)

        fig_csd = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i, c in zip(table1_csd_plot['Income Category'].unique(), colors):
            plot_df_frag = table1_csd_plot.loc[table1_csd_plot['Income Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 1)

        # fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='relative', plot_bgcolor='#F8F9F9', title = f'2026 Projected Households -<br>{geo}', legend_title = "Income Category")

        col_list = []

        for i in table1.columns:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        table1_c, table1_csd_plot_c = projections_2026_hh_size(geo_c, True)

        for i, c in zip(table1_csd_plot_c['Income Category'].unique(), colors):
            plot_df_frag = table1_csd_plot_c.loc[table1_csd_plot_c['Income Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                showlegend = False,
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_csd.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='relative', plot_bgcolor='#F8F9F9', title = f'2026 Projected Households', legend_title = "Income Category")
        fig_csd.update_yaxes(range=[0, max(table1_csd_plot.groupby('HH Category')['value'].sum().max(), table1_csd_plot_c.groupby('HH Category')['value'].sum().max())+10000])
        fig_csd.update_xaxes(fixedrange = True)
        fig_csd.update_yaxes(fixedrange = True)

        table1_j = table1.merge(table1_c, how = 'left', on = 'HH Income Category')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_csd






# new plot 3 for projection

income_col_list = ['20% or under of area median household income (AMHI)', 
                    '21% to 50% of AMHI', 
                    '51% to 80% of AMHI', 
                    '81% to 120% of AMHI', 
                    '121% or over of AMHI']

pp_list = ['1pp', '2pp', '3pp', '4pp', '5pp']


def projections_2026_deltas(geo, IsComparison):
    
    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, 'Geo_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")


    income_l = []
    pp_l = []
    result_csd_l = []


    for i in income_col_list:
        for p in pp_list:
            col_format = f'2026 Population Delta {p} HH with income {i}'
            income_l.append(i)
            pp_l.append(p)
            result_csd_l.append(updated_csd_filtered[col_format].tolist()[0])

    income_l = ['Very Low Income'] * 5 + ['Low Income'] * 5 + ['Moderate Income'] * 5 + ['Median Income'] * 5 + ['High Income'] * 5

    table3 = pd.DataFrame({'Income Category': income_l, 'HH Category': pp_l, 'CSD_Projection': np.round(result_csd_l,0)})
    table3 = table3.fillna(0)
    # table3_csd = table3.pivot_table(values='CSD_Projection', index=['Income Category'], columns=['HH Category'], sort = False)
    table3_csd = table3.pivot_table(values='CSD_Projection', index=['HH Category'], columns=['Income Category'], sort = False)
    table3_csd = table3_csd.reset_index()
    table3_csd = table3_csd.replace([np.inf, -np.inf], 0)
    table3_csd['HH Category'] = ['1 person', '2 people', '3 people', '4 people', '5+ people'][:len(table3_csd['HH Category'])]
    # print(table3_csd['HH Category'])

    table3_csd_plot = table3_csd.replace([np.inf, -np.inf], 0)
#     table3_csd_plot = pd.melt(table3_csd_plot, id_vars = 'Income Category', value_vars = ['1pp', '2pp', '3pp', '4pp', '5pp'])
    table3_csd_plot = pd.melt(table3_csd_plot, id_vars = 'HH Category', value_vars = ['Very Low Income', 'Low Income', 'Moderate Income',
           'Median Income', 'High Income'])
    # table3_csd = table3_csd.replace([np.inf, -np.inf], 0)
    row_total_csd = table3_csd.sum(axis=0)
    row_total_csd[0] = 'Total'
    table3_csd.loc[5, :] = row_total_csd

    if IsComparison != True:
        table3_csd.columns = ['HH Income Category', 'Very Low Income', 'Low Income', 'Moderate Income',
       'Median Income', 'High Income']
        table3_csd['Total'] = table3_csd.sum(axis=1)

    else:
        table3_csd.columns = ['HH Income Category', 'Very Low Income ', 'Low Income ', 'Moderate Income ',
       'Median Income ', 'High Income ']
        table3_csd['Total '] = table3_csd.sum(axis=1)
        # table3_csd = table3_csd.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })

    return table3_csd, table3_csd_plot

@callback(
    Output('datatable7-interactivity', 'columns'),
    Output('datatable7-interactivity', 'data'),
    Output('datatable7-interactivity', 'style_data_conditional'),
    Output('datatable7-interactivity', 'style_cell_conditional'),
    Output('graph11', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('area-scale-store', 'data'),
    Input('datatable7-interactivity', 'selected_columns'),
)
def update_geo_figure8(geo, geo_c, scale, selected_columns):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == scale:
            geo = geo
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        table1, table1_csd_plot = projections_2026_deltas(geo, False)

        fig_csd = go.Figure()
        for i, c in zip(table1_csd_plot['Income Category'].unique(), colors):
            plot_df_frag = table1_csd_plot.loc[table1_csd_plot['Income Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ))

        fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='relative', plot_bgcolor='#F8F9F9', title = f'2026 Projected Households -<br>{geo}', legend_title = "Income Category")
        fig_csd.update_xaxes(fixedrange = True)
        fig_csd.update_yaxes(fixedrange = True)

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_csd


    else:

        if "to-geography-1" == scale:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        table1, table1_csd_plot = projections_2026_deltas(geo, False)

        fig_csd = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i, c in zip(table1_csd_plot['Income Category'].unique(), colors):
            plot_df_frag = table1_csd_plot.loc[table1_csd_plot['Income Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 1)

        # fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='relative', plot_bgcolor='#F8F9F9', title = f'Community 2026 HH -<br>{geo}', legend_title = "Income Category")



        col_list = []

        for i in table1.columns:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        table1_c, table1_csd_plot_c = projections_2026_deltas(geo_c, True)

        for i, c in zip(table1_csd_plot_c['Income Category'].unique(), colors):
            plot_df_frag = table1_csd_plot_c.loc[table1_csd_plot_c['Income Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                showlegend = False,
                hovertemplate= '%{x}, ' + f'{i} - ' + '%{y}<extra></extra>'
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_csd.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='relative', plot_bgcolor='#F8F9F9', title = f'2026 Projected Households', legend_title = "Income Category")
        fig_csd.update_yaxes(range=[min(table1_csd_plot.groupby('HH Category')['value'].sum().min(), table1_csd_plot_c.groupby('HH Category')['value'].sum().min())-100, max(table1_csd_plot.groupby('HH Category')['value'].sum().max(), table1_csd_plot_c.groupby('HH Category')['value'].sum().max())+100])
        fig_csd.update_xaxes(fixedrange = True)
        fig_csd.update_yaxes(fixedrange = True)

        table1_j = table1.merge(table1_c, how = 'left', on = 'HH Income Category')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_csd



# new plot 4 for projection

def projections_2026_pop_income(geo, IsComparison):

    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, 'Geo_Code'].tolist()[0]
    geo_region_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, 'Region_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")
    updated_cd_filtered = updated_cd.query('Geo_Code ==' +  f"{geo_region_code_clicked}")
    updated_csd_filtered

    income_categories_g11 = [
        'income 20% or under of area median household income (AMHI)',
        'income 21% to 50% of AMHI',
        'income 51% to 80% of AMHI',
        'income 81% to 120% of AMHI',
        'income 121% or over of AMHI'
    ]

    pop_2016 = []
    gr_csd = []
    gr_cd = []
    delta = []

    i_l = [
        'Very low Income',
        'Low Income',
        'Moderate Income',
        'Median Income',
        'High Income'
        ]

    for i in income_categories_g11:
        p = updated_csd_filtered[f'Total - Private households by household type including census family structure -   Households with {i} - Total - Household size'].tolist()[0]
        g = updated_csd_filtered[f'2026 Population Growth Rate with {i}'].tolist()[0]
        g_cd = updated_cd_filtered[f'2026 Population Growth Rate with {i}'].tolist()[0]
        d = updated_csd_filtered[f'2026 Population Delta with {i}'].tolist()[0]
        pop_2016.append(p)
        gr_csd.append(g)
        gr_cd.append(g_cd)
        delta.append(d)

    table = pd.DataFrame({'Income Category':  i_l, '2016 Pop.': pop_2016, 'Muni. Growth (%)': gr_csd, 'Regional Growth (%)': gr_cd, 'Delta(Muni. GR)': np.round(delta, 0)})
    table['Delta(Regional GR)'] = np.round(table['2016 Pop.'] * table['Regional Growth (%)'], 0)
    table['2026 Pop.(Muni.)'] = np.round(table['2016 Pop.'] + (table['2016 Pop.'] * table['Muni. Growth (%)']), 0)
    table['2026 Pop.(Regional)'] = np.round(table['2016 Pop.'] + (table['2016 Pop.'] * table['Regional Growth (%)']), 0)

    plot_df1 = table[['Income Category', '2016 Pop.', 'Delta(Muni. GR)']]
    plot_df1.columns = ['Income Category', '2016 Pop.', '2026 Delta']
    plot_df1 = plot_df1.melt(id_vars = 'Income Category', value_vars = ['2016 Pop.', '2026 Delta'])
    plot_df1['Geo'] = 'Muni'

    plot_df2 = table[['Income Category', '2016 Pop.', 'Delta(Regional GR)']]
    plot_df2.columns = ['Income Category', '2016 Pop.', '2026 Delta']
    plot_df2 = plot_df2.melt(id_vars = 'Income Category', value_vars = ['2016 Pop.', '2026 Delta'])
    plot_df2['Geo'] = 'Region'

    plot_df = pd.concat([plot_df1, plot_df2])

    table = table.drop(columns = ['Delta(Muni. GR)', 'Delta(Regional GR)'])
    table['Muni. Growth (%)'] = np.round(table['Muni. Growth (%)']*100,1).astype(str) + '%'
    table['Regional Growth (%)'] = np.round(table['Regional Growth (%)']*100,1).astype(str) + '%'

    if IsComparison == True:
        # table.columns = ['Income Category', '2016 Pop. ', 'Muni. Growth (%) ',
        #    'Regional Growth (%) ', 'Delta(Muni. GR) ', 'Delta(Regional GR) ',
        #    '2026 Pop.(Muni. GR) ', '2026 Pop.(Regional GR) ']
    
        table.columns = ['HH Income Category', '2016 HHs ', 'Muni. Growth Rate (%) ',
           'Regional Growth Rate (%) ', '2026 HHs (Muni. Rate) ', '2026 HHs (Region. Rate) ']
        
    else:
        table.columns = ['HH Income Category', '2016 HHs', 'Muni. Growth Rate (%)',
           'Regional Growth Rate (%)', '2026 HHs (Muni. Rate)', '2026 HHs (Region. Rate)']

    return table, plot_df

@callback(
    Output('datatable8-interactivity', 'columns'),
    Output('datatable8-interactivity', 'data'),
    Output('datatable8-interactivity', 'style_data_conditional'),
    Output('datatable8-interactivity', 'style_cell_conditional'),
    Output('graph12', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('area-scale-store', 'data'),
    Input('datatable8-interactivity', 'selected_columns'),
)
def update_geo_figure8(geo, geo_c, scale, selected_columns):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == scale:
            geo = geo
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        table1, plot_df = projections_2026_pop_income(geo, True)

        fig_pgr = go.Figure()

        for s, c in zip(plot_df['variable'].unique(), colors[3:]):
            
            plot_df_frag = plot_df.loc[plot_df['variable'] == s, :]
            plot_df_frag['Income Category'] = ['Very Low', 'Low', 'Moderate', 'Median', 'High'] * 2
            x = [
                plot_df_frag['Income Category'],
                plot_df_frag['Geo']
            ]
            
            fig_pgr.add_trace(go.Bar(
                x = x,
                y = plot_df_frag['value'],
                name = s,
                marker_color = c,
                hovertemplate= '%{x}, ' + f'{s} - ' + '%{y}<extra></extra>'
            ))

        fig_pgr.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode = "relative", plot_bgcolor='#F8F9F9', title = f'2026 Household Projections -<br>{geo}', legend_title = "Population")
        fig_pgr.update_xaxes(fixedrange = True)
        fig_pgr.update_yaxes(fixedrange = True)

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_pgr


    else:

        if "to-geography-1" == scale:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        table1, plot_df = projections_2026_pop_income(geo, True)


        fig_pgr = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for s, c in zip(plot_df['variable'].unique(), colors[3:]):
            
            plot_df_frag = plot_df.loc[plot_df['variable'] == s, :]
            plot_df_frag['Income Category'] = ['Very Low', 'Low', 'Moderate', 'Median', 'High'] * 2

            x = [
                plot_df_frag['Income Category'],
                plot_df_frag['Geo']
            ]
            
            fig_pgr.add_trace(go.Bar(
                x = x,
                y = plot_df_frag['value'],
                name = s,
                marker_color = c,
                hovertemplate= '%{x}, ' + f'{s} - ' + '%{y}<extra></extra>'

            ),row = 1, col = 1)

        col_list = []

        for i in table1.columns:
            if i == 'Income Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        table1_c, plot_df_c = projections_2026_pop_income(geo_c, False)

        for s, c in zip(plot_df_c['variable'].unique(), colors[3:]):
            
            plot_df_frag = plot_df_c.loc[plot_df_c['variable'] == s, :]
            plot_df_frag['Income Category'] = ['Very Low', 'Low', 'Moderate', 'Median', 'High'] * 2

            x = [
                plot_df_frag['Income Category'],
                plot_df_frag['Geo']
            ]
            
            fig_pgr.add_trace(go.Bar(
                x = x,
                y = plot_df_frag['value'],
                name = s,
                marker_color = c,
                hovertemplate= '%{x}, ' + f'{s} - ' + '%{y}<extra></extra>',
                showlegend = False
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_pgr.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode = "relative", plot_bgcolor='#F8F9F9', title = f'2026 Household Projections', legend_title = "Population")
        fig_pgr.update_yaxes(range=[0, max(plot_df.groupby(['Income Category', 'Geo'])['value'].sum().max(), plot_df_c.groupby(['Income Category', 'Geo'])['value'].sum().max())+10000])
        fig_pgr.update_xaxes(fixedrange = True)
        fig_pgr.update_yaxes(fixedrange = True)

        table1_j = table1.merge(table1_c, how = 'left', on = 'HH Income Category')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_pgr
    

# new plot 5 for projection

def projections_2026_pop_hh(geo, IsComparison):

    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, 'Geo_Code'].tolist()[0]
    geo_region_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, 'Region_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")
    updated_cd_filtered = updated_cd.query('Geo_Code ==' +  f"{geo_region_code_clicked}")
    updated_csd_filtered

    hh_category = [
        '1pp',
        '2pp',
        '3pp',
        '4pp',
        '5pp'
    ]

    pop_2016 = []
    gr_csd = []
    gr_cd = []
    delta = []

    h_l = ['1 Person', '2 People', '3 People', '4 People', '5+ People']

    for i in hh_category:
        p = updated_csd_filtered[f'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   {i}'].tolist()[0]
        g = updated_csd_filtered[f'2026 Population Growth Rate {i} HH'].tolist()[0]
        g_cd = updated_cd_filtered[f'2026 Population Growth Rate {i} HH'].tolist()[0]
        d = updated_csd_filtered[f'2026 Population Delta {i} HH'].tolist()[0]
        pop_2016.append(p)
        gr_csd.append(g)
        gr_cd.append(g_cd)
        delta.append(d)

    table = pd.DataFrame({'HH Category':  h_l, '2016 Pop.': pop_2016, 'Muni. Growth (%)': gr_csd, 'Regional Growth (%)': gr_cd, 'Delta(Muni. GR)': np.round(delta, 0)})
    table['Delta(Regional GR)'] = np.round(table['2016 Pop.'] * table['Regional Growth (%)'], 0)
    table['2026 Pop.(Muni.)'] = np.round(table['2016 Pop.'] + (table['2016 Pop.'] * table['Muni. Growth (%)']), 0)
    table['2026 Pop.(Regional)'] = np.round(table['2016 Pop.'] + (table['2016 Pop.'] * table['Regional Growth (%)']), 0)

    table

    plot_df1 = table[['HH Category', '2016 Pop.', 'Delta(Muni. GR)']]
    plot_df1.columns = ['HH Category', '2016 Pop.', '2026 Delta']
    plot_df1 = plot_df1.melt(id_vars = 'HH Category', value_vars = ['2016 Pop.', '2026 Delta'])
    plot_df1['Geo'] = 'Muni'

    plot_df2 = table[['HH Category', '2016 Pop.', 'Delta(Regional GR)']]
    plot_df2.columns = ['HH Category', '2016 Pop.', '2026 Delta']
    plot_df2 = plot_df2.melt(id_vars = 'HH Category', value_vars = ['2016 Pop.', '2026 Delta'])
    plot_df2['Geo'] = 'Region'

    plot_df = pd.concat([plot_df1, plot_df2])

    table = table.drop(columns = ['Delta(Muni. GR)', 'Delta(Regional GR)'])
    table['Muni. Growth (%)'] = np.round(table['Muni. Growth (%)']*100,1).astype(str) + '%'
    table['Regional Growth (%)'] = np.round(table['Regional Growth (%)']*100,1).astype(str) + '%'

    if IsComparison == True:
        # table.columns = ['Income Category', '2016 Pop. ', 'Muni. Growth (%) ',
        #    'Regional Growth (%) ', 'Delta(Muni. GR) ', 'Delta(Regional GR) ',
        #    '2026 Pop.(Muni. GR) ', '2026 Pop.(Regional GR) ']
    
        table.columns = ['HH Size', '2016 HHs ', 'Muni. Growth Rate (%) ',
           'Regional Growth Rate (%) ', '2026 HHs (Muni. Rate) ', '2026 HHs (Region. Rate) ']

    else:
        table.columns = ['HH Size', '2016 HHs', 'Muni. Growth Rate (%)',
           'Regional Growth Rate (%)', '2026 HHs (Muni. Rate)', '2026 HHs (Region. Rate)']

    return table, plot_df

@callback(
    Output('datatable9-interactivity', 'columns'),
    Output('datatable9-interactivity', 'data'),
    Output('datatable9-interactivity', 'style_data_conditional'),
    Output('datatable9-interactivity', 'style_cell_conditional'),
    Output('graph13', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('area-scale-store', 'data'),
    Input('datatable9-interactivity', 'selected_columns'),
)
def update_geo_figure9(geo, geo_c, scale, selected_columns):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == scale:
            geo = geo
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        table1, plot_df = projections_2026_pop_hh(geo, True)

        fig_pgr = go.Figure()

        for s, c in zip(plot_df['variable'].unique(), colors[3:]):
            
            plot_df_frag = plot_df.loc[plot_df['variable'] == s, :]

            x = [
                plot_df_frag['HH Category'],
                plot_df_frag['Geo']
            ]
            
            fig_pgr.add_trace(go.Bar(
                x = x,
                y = plot_df_frag['value'],
                name = s,
                marker_color = c,
                hovertemplate= '%{x}, ' + f'{s} - ' + '%{y}<extra></extra>'
            ))

        fig_pgr.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode = "relative", plot_bgcolor='#F8F9F9', title = f'2026 Household Projections -<br>{geo}', legend_title = "Population")
        fig_pgr.update_xaxes(fixedrange = True)
        fig_pgr.update_yaxes(fixedrange = True)

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_pgr


    else:

        if "to-geography-1" == scale:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == scale:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        table1, plot_df = projections_2026_pop_hh(geo, True)


        fig_pgr = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for s, c in zip(plot_df['variable'].unique(), colors[3:]):
            
            plot_df_frag = plot_df.loc[plot_df['variable'] == s, :]
            plot_df_frag['HH Category'] = ['1 Person', '2 People', '3 People', '4 People', '5+ People'] * 2

            x = [
                plot_df_frag['HH Category'],
                plot_df_frag['Geo']
            ]
            
            fig_pgr.add_trace(go.Bar(
                x = x,
                y = plot_df_frag['value'],
                name = s,
                marker_color = c,
                hovertemplate= '%{x}, ' + f'{s} - ' + '%{y}<extra></extra>'

            ),row = 1, col = 1)

        col_list = []

        for i in table1.columns:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        table1_c, plot_df_c = projections_2026_pop_hh(geo_c, False)

        for s, c in zip(plot_df_c['variable'].unique(), colors[3:]):
            
            plot_df_frag = plot_df_c.loc[plot_df_c['variable'] == s, :]
            plot_df_frag['HH Category'] = ['1 Person', '2 People', '3 People', '4 People', '5+ People'] * 2

            x = [
                plot_df_frag['HH Category'],
                plot_df_frag['Geo']
            ]
            
            fig_pgr.add_trace(go.Bar(
                x = x,
                y = plot_df_frag['value'],
                name = s,
                marker_color = c,
                hovertemplate= '%{x}, ' + f'{s} - ' + '%{y}<extra></extra>',
                showlegend = False
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'HH Category':
                col_list.append({"name": ["Areas", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_pgr.update_layout(modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode = "relative", plot_bgcolor='#F8F9F9', title = f'2026 Household Projections', legend_title = "Population")
        fig_pgr.update_yaxes(range=[0, max(plot_df.groupby(['HH Category', 'Geo'])['value'].sum().max(), plot_df_c.groupby(['HH Category', 'Geo'])['value'].sum().max())+10000])
        fig_pgr.update_xaxes(fixedrange = True)
        fig_pgr.update_yaxes(fixedrange = True)

        table1_j = table1.merge(table1_c, how = 'left', on = 'HH Size')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'font_size': comparison_font_size,
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_pgr
    

