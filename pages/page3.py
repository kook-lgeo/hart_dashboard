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

df_income = pd.read_sql_table('income', engine.connect())
# df_income = pd.read_csv("./sources/income.csv")

# Importing partners data

df_partners = pd.read_sql_table('partners', engine.connect())
#df_partners = pd.read_csv("./sources/partners_small.csv")

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

# Importing Province Boundaries shape data


gdf_p_code_added = gpd.read_file('./sources/mapdata_simplified/province.shp')
gdf_p_code_added = gdf_p_code_added.set_index('Geo_Code')

# Importing subregions which don't have data

not_avail = pd.read_csv('not_in_list.csv')
not_avail['CSDUID'] = not_avail['CSDUID'].astype(str)

# Configuration for plot icons

config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['zoom', 'lasso2d', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',]}

# Preprocessing

income_category = df_income.drop(['Geography'], axis=1)

income_category = income_category.rename(columns = {'Formatted Name': 'Geography'})

joined_df = income_category.merge(df_partners, how = 'left', on = 'Geography')

joined_df_filtered = joined_df.query('Geography == "Fraser Valley (CD, BC)"')

x_base =['Very Low Income',
            'Low Income',
            'Moderate Income',
            'Median Income',
            'High Income',
            ]

x_columns = ['Rent 20% of AMHI',
             'Rent 50% of AMHI',
             'Rent 80% of AMHI',
             'Rent 120% of AMHI',
             'Rent 120% of AMHI'
            ]

hh_p_num_list = [1,2,3,4,'5 or more']

amhi_range = ['20% or under of AMHI', '21% to 50% of AMHI', '51% to 80% of AMHI', '81% to 120% of AMHI', '121% and more of AMHI']

income_ct = [x + f" ({a})" for x, a in zip(x_base, amhi_range)]

x_list = []

i = 0
for b, c in zip(x_base, x_columns):
    if i < 4:
        x = b + " ($" + str(joined_df_filtered[c].tolist()[0]) + ")"
        x_list.append(x)
    else:
        x = b + " (>$" + str(joined_df_filtered[c].tolist()[0]) + ")"
        x_list.append(x)
    i += 1

columns = ['Percent HH with income 20% or under of AMHI in core housing need',
            'Percent HH with income 21% to 50% of AMHI in core housing need',
            'Percent HH with income 51% to 80% of AMHI in core housing need',
            'Percent HH with income 81% to 120% of AMHI in core housing need',
            'Percent HH with income 121% or more of AMHI in core housing need'
            ]

plot_df = pd.DataFrame({'Income_Category': x_list, 'Percent HH': joined_df_filtered[columns].T.iloc[:,0].tolist()})

# colors = ['#D7F3FD', '#B0E6FC', '#88D9FA', '#61CDF9', '#39C0F7']
# hh_colors = ['#D8EBD4', '#B7DCAE', '#93CD8A', '#6EC067', '#3DB54A']
colors = ['#D7F3FD', '#88D9FA', '#39C0F7', '#099DD7', '#044762']
hh_colors = ['#D8EBD4', '#93CD8A', '#3DB54A', '#297A32', '#143D19']
hh_type_color = ['#3949CE', '#3EB549', '#39C0F7']
columns_color_fill = ['#F3F4F5', '#EBF9FE', '#F0FAF1']
map_colors_wo_black = ['#39C0F7', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#F4F739']
map_colors_w_black = ['#000000', '#39C0F7', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#F4F739']
modebar_color = '#099DD7'
modebar_activecolor = '#044762'

fig = go.Figure()
for i, c in zip(plot_df['Income_Category'], colors):
    plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
    fig.add_trace(go.Bar(
        x = plot_df_frag['Income_Category'],
        y = plot_df_frag['Percent HH'],
        name = i,
        marker_color = c,
        hovertemplate= '%{x} - ' + '%{y:.3s}<extra></extra>'
    ))
fig.update_layout(plot_bgcolor='#F8F9F9', title = 'Percent HH By Geography', legend_title = "Income")

table = joined_df_filtered[['Rent 20% of AMHI', 'Rent 50% of AMHI']]
table2 = joined_df_filtered[['Rent 20% of AMHI', 'Rent 50% of AMHI']]
fig5 = fig
fig6 = fig
fig7 = fig


gdf_p_code_added["rand"] = np.random.randint(1, 100, len(gdf_p_code_added))

fig_m = go.Figure()

fig_m.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_p_code_added.geometry.to_json()), 
                                locations = gdf_p_code_added.index, 
                                z = gdf_p_code_added.rand, 
                                showscale = False, 
                                hovertext= gdf_p_code_added.NAME,
                                marker = dict(opacity = 0.4),
                                marker_line_width=.5))


fig_m.update_layout(mapbox_style="carto-positron",
                mapbox_center = {"lat": gdf_p_code_added.geometry.centroid.y.mean()+10, "lon": gdf_p_code_added.geometry.centroid.x.mean()},
                height = 500,
                width = 1000,
                mapbox_zoom = 1.4,
                autosize=True)



# Setting layout for dashboard

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
# server = app.server

# dash.register_page(__name__)

layout = html.Div(children = [
        dcc.Store(id='main-area', storage_type='local'),
        dcc.Store(id='comparison-area', storage_type='local'),
        html.Div(
        children = [
            html.Div([
                html.H2(children = html.Strong("Projections"), id = 'home')
            ]),


        # 2026 Projections by HH Size and Income Level

            html.H3(children = html.Strong('2026 Projections by HH Size and Income Level'), id = 'overview-scenario8'),

            # Table

            html.Div([
                html.Div([

                    html.H5(children = html.Strong('Community 2026 HH'), className = 'table-title'),

                    dash_table.DataTable(
                        id='datatable3-interactivity',
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
                        merge_duplicate_headers=True,
                        style_cell = {'font-family': 'Bahnschrift'},
                        export_format="csv",
                        # style_table={'minWidth': '100%'},
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable3-interactivity-container'),

                    # Graphs

                    html.Div(children = [ 
                        html.Div(
                            dcc.Graph(
                                id='graph7',
                                figure=fig5,
                                config = config,
                            ),
                            style={'width': '100%', 'display': 'inline-block'}
                        ),
                    ]
                    ),

                ], className = 'csd_table_plot', style={'width': '65%', 'display': 'inline-block'}),

                html.Div([
                   
                    html.H5(children = html.Strong('Community 2026 HH (Regional Rates)'), className = 'table-title'),

                    dash_table.DataTable(
                        id='datatable4-interactivity',
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
                    html.Div(id='datatable4-interactivity-container'),
                # ], className = 'tables', style={'width': '70%', 'display': 'inline-block'}),

                    # Graphs

                    html.Div(children = [ 
                        html.Div(
                            dcc.Graph(
                                id='graph8',
                                figure=fig5,
                                config = config,
                            ),
                            style={'width': '100%', 'display': 'inline-block'}
                        ),
                    ]
                    ),

            ], className = 'cdr_table_plot', style={'width': '65%', 'display': 'inline-block'}),
        ]),


        ], className = 'dashboard'
    ), 
], className = 'background'#style = {'backgroud-color': '#fffced'}
)




# 2026 Projections by HH Size and Income Level

income_col_list = ['20% or under of area median household income (AMHI)', 
                    '21% to 50% of AMHI', 
                    '51% to 80% of AMHI', 
                    '81% to 120% of AMHI', 
                    '121% or over of AMHI']

pp_list = ['1pp', '2pp', '3pp', '4pp', '5pp']

income_col_list_r = ['20%orunderofareamedianhouseholdincome(AMHI)', 
                    '21%to50%ofAMHI', 
                    '51%to80%ofAMHI', 
                    '81%to120%ofAMHI', 
                    '121%oroverofAMHI']

pp_list_r = ['1person', '2persons', '3persons', '4persons', '5ormorepersonshousehold']

def projections_2026(geo, IsComparison):
    
    df_csd_proj_merged_filtered = df_csd_proj_merged.loc[df_csd_proj_merged['Geography'] == geo,:]
    geo_region = int(mapped_geo_code.loc[mapped_geo_code['Geography'] == geo]['Region_Code'].values[0])
    df_cd_grow_merged_filtered = df_cd_grow_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region,:]

    geo_region_name = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]

    income_l = []
    pp_l = []
    result_csd_l = []
    result_g_l = []
    result_t_l = []

    for i, i_r in zip(income_col_list, income_col_list_r):
        for p, p_r in zip(pp_list, pp_list_r):
            col_format_r = f'TotalPrivatehouseholdsbyhouseholdtypeincludingcensusfamilystructure - Householdswithincome{i_r} - {p_r} - 2016'
            col_format_p = f'2026 Projected {p} HH with income {i}'
            col_format_g = f'2026 Projected Growth {p} HH with income {i}'
            income_l.append(i)
            pp_l.append(p)
            result_csd_l.append(df_csd_proj_merged_filtered[col_format_p].tolist()[0])
            result_t_l.append(df_csd_proj_merged_filtered[col_format_r].tolist()[0])
            result_g_l.append(df_cd_grow_merged_filtered[col_format_g].tolist()[0])

    income_l = ['Very Low Income'] * 5 + ['Low Income'] * 5 + ['Moderate Income'] * 5 + ['Median Income'] * 5 + ['High Income'] * 5

    table3 = pd.DataFrame({'Income_Category': income_l, 'HH_Category': pp_l, 'CSD_Projection': result_csd_l, 'CSD_Total': result_t_l, 'Growth': result_g_l})
    table3 = table3.fillna(0)
    table3['CSD_Total'] = table3['CSD_Total'].astype(float)
    table3['Growth'] = table3['Growth'].astype(float)
    table3['CSD_Projection'] = np.round(table3['CSD_Projection'].astype(float), -1)
    table3['Projection'] = np.round((table3['CSD_Total'] * table3['Growth']) + table3['CSD_Total'], -1)

    table3_csd = table3.pivot_table(values='CSD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
    table3_csd = table3_csd.reset_index()

    table3_cd_r = table3.pivot_table(values='Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
    table3_cd_r = table3_cd_r.reset_index()

    table3_csd_plot = table3_csd.replace([np.inf, -np.inf], 0)
    table3_csd_plot = pd.melt(table3_csd_plot, id_vars = 'Income_Category', value_vars = ['1pp', '2pp', '3pp', '4pp', '5pp'])

    table3_cd_r_plot = table3_cd_r.replace([np.inf, -np.inf], 0)
    table3_cd_r_plot = pd.melt(table3_cd_r_plot, id_vars = 'Income_Category', value_vars = ['1pp', '2pp', '3pp', '4pp', '5pp'])

    table3_csd = table3_csd.replace([np.inf, -np.inf], 0)
    row_total_csd = table3_csd.sum(axis=0)
    row_total_csd[0] = 'Total'
    table3_csd.loc[5, :] = row_total_csd

    table3_cd_r = table3_cd_r.replace([np.inf, -np.inf], 0)
    row_total_cd_r = table3_cd_r.sum(axis=0)
    row_total_cd_r[0] = 'Total'
    table3_cd_r.loc[5, :] = row_total_cd_r

    if IsComparison != True:
        table3_csd['Total'] = table3_csd.sum(axis=1)
        table3_cd_r['Total'] = table3_cd_r.sum(axis=1)
    else:
        table3_csd['Total '] = table3_csd.sum(axis=1)
        table3_cd_r['Total '] = table3_cd_r.sum(axis=1) 
        table3_csd = table3_csd.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })
        table3_cd_r = table3_cd_r.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })
               
        
    return table3_csd, table3_cd_r, table3_csd_plot, table3_cd_r_plot, geo_region_name


@callback(
    Output('datatable3-interactivity', 'columns'),
    Output('datatable3-interactivity', 'data'),
    Output('datatable3-interactivity', 'style_data_conditional'),
    Output('datatable3-interactivity', 'style_cell_conditional'),
    Output('datatable4-interactivity', 'columns'),
    Output('datatable4-interactivity', 'data'),
    Output('datatable4-interactivity', 'style_data_conditional'),
    Output('datatable4-interactivity', 'style_cell_conditional'),
    Output('graph7', 'figure'),
    Output('graph8', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('datatable3-interactivity', 'selected_columns'),
    Input('datatable4-interactivity', 'selected_columns'),
)
def update_table3(geo, geo_c, selected_columns, selected_columns2):
    
    # print(geo, geo_c)

    clicked_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, :]['Geo_Code'].tolist()[0]

    if geo_c != None:
        clicked_code_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c, :]['Geo_Code'].tolist()[0]
    else:
        clicked_code_c = 0

    # If Census Division or Province is selected

    if len(str(clicked_code)) < 6:

        table3_csd = pd.DataFrame({'Not Available in CD/Province level. Please select CSD level region':[0]})
        table3_cd_r = pd.DataFrame({'Not Available in CD/Province level. Please select CSD level region':[0]})
        
        col_list_csd = []

        for i in table3_csd.columns:
            col_list_csd.append({"name": [i],
                                    "id": i, })

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            col_list_cd_r.append({"name": [i],
                                    "id": i, })

        style_cell_conditional_csd=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_csd.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_csd.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        style_cell_conditional_cd_r=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_cd_r.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_cd_r.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        fig_csd = px.line(x = ['Not Available in CD/Province level. Please select CSD level region'], y = ['Not Available in CD/Province level. Please select CSD level region'])
        fig_cd_r = px.line(x = ['Not Available in CD/Province level. Please select CSD level region'], y = ['Not Available in CD/Province level. Please select CSD level region'])

        return col_list_csd, \
                table3_csd.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                style_cell_conditional_csd, \
                col_list_cd_r, \
                table3_cd_r.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                style_cell_conditional_cd_r, fig_csd, fig_cd_r
    

    # If Census SubDivision is selected

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None) or len(str(clicked_code_c)) < 6:

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'

        table3_csd, table3_cd_r, table3_csd_plot, table3_cd_r_plot, geo_region_name = projections_2026(geo, IsComparison = False)

        fig_csd = go.Figure()
        for i, c in zip(table3_csd_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_csd_plot.loc[table3_csd_plot['HH_Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ))

        fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026 HH - {geo}', legend_title = "Income Category")


        fig_cd_r = go.Figure()
        for i, c in zip(table3_cd_r_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_cd_r_plot.loc[table3_cd_r_plot['HH_Category'] == i, :]
            fig_cd_r.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ))

        fig_cd_r.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026 HH (Regional Rates) - {geo_region_name}', legend_title = "Income Category")

        col_list_csd = []

        for i in table3_csd.columns:
            col_list_csd.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            col_list_cd_r.append({"name": [geo_region_name, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        style_cell_conditional_csd=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_csd.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_csd.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        style_cell_conditional_cd_r=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_cd_r.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_cd_r.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        return col_list_csd, \
                table3_csd.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                style_cell_conditional_csd, \
                col_list_cd_r, \
                table3_cd_r.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                style_cell_conditional_cd_r, fig_csd, fig_cd_r

    else:


        table3_csd, table3_cd_r, table3_csd_plot, table3_cd_r_plot, geo_region_name = projections_2026(geo, IsComparison = False)
        
        # Comparison Tables/Plots
        
        table3_csd_c, table3_cd_r_c, table3_csd_c_plot, table3_cd_r_c_plot, geo_region_name_c = projections_2026(geo_c, IsComparison = True)

        fig_csd = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        n = 0
        for i, c in zip(table3_csd_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_csd_plot.loc[table3_csd_plot['HH_Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
            ), row = 1, col = 1)
            n += 1

        n = 0
        for i, c in zip(table3_csd_c_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_csd_c_plot.loc[table3_csd_c_plot['HH_Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
                showlegend = False
            ), row = 1, col = 2)
            n += 1
            
        fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026', legend_title = "HH Category")
        fig_csd.update_yaxes(range=[0, max(table3_csd_plot.groupby('Income_Category')['value'].sum().max(), table3_csd_c_plot.groupby('Income_Category')['value'].sum().max())+5000])

        fig_cd_r = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo_region_name}", f"{geo_region_name_c}"), shared_yaxes=True, shared_xaxes=True)

        n = 0
        for i, c in zip(table3_cd_r_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_cd_r_plot.loc[table3_cd_r_plot['HH_Category'] == i, :]
            fig_cd_r.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
            ), row = 1, col = 1)
            n += 1

        n = 0
        for i, c in zip(table3_cd_r_c_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_cd_r_c_plot.loc[table3_cd_r_c_plot['HH_Category'] == i, :]
            fig_cd_r.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
                showlegend = False
            ), row = 1, col = 2)
            n += 1
            
        fig_cd_r.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026 HH (Regional Rates)', legend_title = "HH Category")
        fig_cd_r.update_yaxes(range=[0, max(table3_cd_r_plot.groupby('Income_Category')['value'].sum().max(), table3_cd_r_c_plot.groupby('Income_Category')['value'].sum().max())+5000])


        col_list_csd = []

        for i in table3_csd.columns:
            if i == 'Income_Category':
                col_list_csd.append({"name": ["Subregions", i], "id": i})
            else:
                col_list_csd.append({"name": [geo, i],
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            if i == 'Income_Category':
                col_list_cd_r.append({"name": ["Regions", i], "id": i})
            else:
                col_list_cd_r.append({"name": [geo_region_name, i],
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        for i in table3_csd_c.columns[1:]:
            if i == 'Income_Category':
                col_list_csd.append({"name": ["Subregions", i], "id": i})
            else:
                col_list_csd.append({"name": [geo_c, i], 
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        for i in table3_cd_r_c.columns[1:]:
            if i == 'Income_Category':
                col_list_cd_r.append({"name": ["Regions", i], "id": i})
            else:
                col_list_cd_r.append({"name": [geo_region_name_c, i], 
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        table3_csd_j = table3_csd.merge(table3_csd_c, how = 'left', on = 'Income_Category')
        table3_cd_r_j = table3_cd_r.merge(table3_cd_r_c, how = 'left', on = 'Income_Category')

        style_cell_conditional_csd=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_csd.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table3_csd_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_csd.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        style_cell_conditional_cd_r=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_cd_r.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table3_cd_r_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_cd_r.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]


        return col_list_csd, \
                table3_csd_j.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                style_cell_conditional_csd, \
                col_list_cd_r, \
                table3_cd_r_j.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                style_cell_conditional_cd_r, fig_csd, fig_cd_r
