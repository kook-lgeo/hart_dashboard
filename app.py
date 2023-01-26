from dash import Dash, dcc, dash_table, html, Input, Output, ctx
import dash
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import warnings
import json
import geopandas as gpd
import fiona
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

# gdf_p = gpd.read_file('./sources/Province Boundaries/Canada.shp')
# gdf_p['NAME'] = gdf_p['NAME'].apply(lambda x: x.replace("Yukon Territory", "Yukon"))
# gdf_p['NAME'] = gdf_p['NAME'].apply(lambda x: x.replace("Quebec", "Québec"))

# df_province_list2 = df_province_list.copy()
# df_province_list2['NAME'] = df_province_list2['Geography'].apply(lambda x: x.replace(" (Province)", ""))

# gdf_p_code_added = gdf_p.merge(df_province_list2[['NAME', 'Geo_Code']], how = 'left', on = 'NAME')
# gdf_p_code_added = gdf_p_code_added.to_crs("EPSG:4326")
# gdf_p_code_added['lat'] = gdf_p_code_added.geometry.centroid.y
# gdf_p_code_added['lon'] = gdf_p_code_added.geometry.centroid.x
# gdf_p_code_added.to_file('./sources/mapdata/province.shp')

gdf_p_code_added = gpd.read_file('./sources/mapdata/province.shp')
gdf_p_code_added = gdf_p_code_added.set_index('Geo_Code')

# Importing Region Boundaries shape data

# gdf_r = gpd.read_file('./sources/Census Divisions/lcd_000a16a_e.shp')
# gdf_r = gdf_r.to_crs("EPSG:4326")
# gdf_r['lat'] = gdf_r.geometry.centroid.y
# gdf_r['lon'] = gdf_r.geometry.centroid.x
# gdf_r.to_file('./sources/mapdata/region.shp')

gdf_r = gpd.read_file('./sources/mapdata/region.shp')
gdf_r = gdf_r.set_index("CDUID")

# Importing SubRegion Boundaries shape data

# gdf_sr = gpd.read_file('./sources/Census SubDivisions/lcsd000b16a_e.shp')
# gdf_sr = gdf_sr.to_crs("EPSG:4326")
# gdf_sr['lat'] = gdf_sr.geometry.centroid.y
# gdf_sr['lon'] = gdf_sr.geometry.centroid.x
# gdf_sr.to_file('./sources/mapdata/subregion.shp')

# gdf_sr = gpd.read_file('./sources/mapdata/subregion.shp')
# gdf_sr = gdf_sr.set_index("CSDUID")

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

colors = ['#0B4952', '#4A8F97', '#4a5b97', '#FAB88A', '#FFDD5D', ]

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
fig.update_layout(plot_bgcolor='#f0faff', title = 'Percent HH By Geography', legend_title = "Income")

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
                                marker = dict(opacity = 0.2),
                                marker_line_width=.5))


fig_m.update_layout(mapbox_style="open-street-map",
                mapbox_center = {"lat": gdf_p_code_added.geometry.centroid.y.mean()+10, "lon": gdf_p_code_added.geometry.centroid.x.mean()},
                height = 500,
                width = 1000,
                mapbox_zoom = 1.4,
                autosize=True)



# Setting layout for dashboard

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(children = [
        html.Div(
        children = [
            html.Div([
                html.H2(children = html.Strong("HART Dashboard"), id = 'home')
            ]),

            # Dropdown for sector selection
            html.H3(children = html.Strong('Area Selection'), id = 'area-selection'),

            # Reset Button for Map

            html.Div(children = [ 

                html.Div(children = [                     
                    html.Button('Reset Map', id='reset-map', n_clicks=0),     
                                    ], className = 'region_button'
                    ),                
                ], 
                style={'width': '55%', 'display': 'inline-block', 'padding-bottom': '20px', 'padding-top': '10px'}
            ),

            # Map

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='canada_map',
                        figure=fig_m
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),

            html.Div(children = [
                html.Strong('Select Area'),
                # dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver A RDA (CSD, BC)', id='all-geo-dropdown'),
                dcc.Dropdown(df_geo_list['Geography'].unique(), 'Greater Vancouver A RDA (CSD, BC)', id='all-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '20px', 'padding-top': '20px'}
            ),

            html.Div(children = [
                html.Strong('Comparison Area'),
                # dcc.Dropdown(joined_df['Geography'].unique(), id='comparison-geo-dropdown'),
                dcc.Dropdown(df_geo_list['Geography'].unique(), id='comparison-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Area Scale Selection

            html.H3(children = html.Strong('Area Scale Selection'), id = 'area-scale'),

            html.Div(children = [ 

                html.Div(children = [                     
                    html.Button('To Subregion', id='to-geography-1', n_clicks=0),     
                                    ], className = 'region_button'
                    ),           
                html.Div(children = [ 
                    html.Button('To Region', id='to-region-1', n_clicks=0),
                                    ], className = 'region_button'
                    ),         
                html.Div(children = [ 
                    html.Button('To Province', id='to-province-1', n_clicks=0),
                                    ], className = 'region_button'
                    ),         
                ], 
                style={'width': '55%', 'display': 'inline-block', 'padding-bottom': '20px', 'padding-top': '10px'}
            ),


        # Percent of Household Size Categories in Core Housing Need, by Area Median Household Income (AMHI)

            html.H3(children = html.Strong('Area Median Household Income (AMHI) Categories and Shelter Costs'), id = 'overview-scenario3'),

            # Table

            html.Div([
                dash_table.DataTable(
                    id='datatable-interactivity',
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                    ],
                    data=table.to_dict('records'),
                    editable=True,
                    # filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable=False,#"multi",
                    row_selectable=False,#"multi",
                    row_deletable=False,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current= 0,
                    page_size= 10,
                    merge_duplicate_headers=True,
                    # export_format = "csv",
                    # style_data = {'font_size': '1.0rem', 'width': '100px'},
                    style_header = {'text-align': 'middle', 'fontWeight': 'bold'}#{'whiteSpace': 'normal', 'font_size': '1.0rem'}
                ),
                html.Div(id='datatable-interactivity-container')
            ], style={'width': '80%', 'padding-top': '30px', 'padding-bottom': '30px', 'display': 'block'}
            ),



        # Percent of Households (HHs) in Core Housing Need, by Household Income Category

            html.H3(children = html.Strong('Percent of Households (HHs) in Core Housing Need, by Household Income Category'), id = 'overview-scenario'),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph',
                        figure=fig
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),


        # Percent of Household Size Categories in Core Housing Need, by Area Median Household Income (AMHI)

            html.H3(children = html.Strong('Percent of Household Size Categories in Core Housing Need, by AMHI'), id = 'overview-scenario2'),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph2',
                        figure=fig
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),

        # Percent of Household Size Categories in Core Housing Need, by Area Median Household Income (AMHI)

            html.H3(children = html.Strong('2016 Affordable Housing Deficit'), id = 'overview-scenario4'),

            # Table
            
            html.Div(children = [ 

                html.Div([
                    dash_table.DataTable(
                        id='datatable2-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": False} for i in table2.columns
                        ],
                        data=table2.to_dict('records'),
                        editable=True,
                        # filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable=False,#"multi",
                        row_selectable=False,#"multi",
                        row_deletable=False,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        merge_duplicate_headers=True,
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'},
                        # export_format = "csv"
                    ),
                    html.Div(id='datatable2-interactivity-container')
                ], style={'width': '80%', 'padding-top': '30px', 'padding-bottom': '30px'}
                ),

            ],style={'width': '80%'}
            ),


        # Percentage of Households (HHs) in Core Housing Need by Priority Population

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population'), id = 'overview-scenario5'),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph5',
                        figure=fig5
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),


        # Percentage of Households (HHs) in Core Housing Need by Priority Population

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population and Income'), id = 'overview-scenario6'),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph6',
                        figure=fig6
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),


        # Percentage of Households (HHs) in Core Housing Need by Priority Population

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population and HH Size'), id = 'overview-scenario7'),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph7',
                        figure=fig7
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),


        # Percent of Household Size Categories in Core Housing Need, by Area Median Household Income (AMHI)

            html.H3(children = html.Strong('2026 Projections by HH Size and Income Level'), id = 'overview-scenario8'),

            # Table

            html.Div([
                html.Div([
                    html.Label(children = html.Strong('Community 2026 HH'), className = 'table-title'),
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
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable3-interactivity-container'),
                ], className = 'tables'),
                html.Div([
                    html.Label(children = html.Strong('Community 2026 HH (Regional Rates)'), className = 'table-title'),
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
                        merge_duplicate_headers=True,
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable4-interactivity-container'),
                ], className = 'tables'),
                html.Div([
                    html.Label(children = html.Strong('Regional 2026 HH'), className = 'table-title'),
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
                        merge_duplicate_headers=True,
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable5-interactivity-container'),
                ], className = 'tables'),
            ], style={'width': '80%', 'padding-top': '30px', 'padding-bottom': '30px', 'display': 'block'}
            ),


            # Raw data download

            html.Div([
            html.Button("Download Full Raw Data", id="ov7-download-csv"),
            dcc.Download(id="ov7-download-text")
            ], 
            style={'width': '12%', 'display': 'inline-block', 'padding-bottom': '50px'}
            ),


        ], className = 'dashboard'
    ), 
], className = 'background'#style = {'backgroud-color': '#fffced'}
)



# Area Selection Map

@app.callback(
    Output('canada_map', 'figure'),
    Output('all-geo-dropdown', 'value'),
    [Input('canada_map', 'clickData')],
    Input('reset-map', 'n_clicks'),
    )
def update_map(clickData, btn1):
    # print(clickData, btn1)
        # if "reset-map" == ctx.triggered_id:
        #     geo = geo
        # elif "to-region-1" == ctx.triggered_id:
        #     geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        # elif "to-province-1" == ctx.triggered_id:
        #     geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]    
    # print(clickData)
    # print(clickData['points'][0]['location'])

    if "reset-map" == ctx.triggered_id:

        gdf_p_code_added["rand"] = np.random.randint(1, 100, len(gdf_p_code_added))

        fig_m = go.Figure()

        fig_m.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_p_code_added.geometry.to_json()), 
                                        locations = gdf_p_code_added.index, 
                                        z = gdf_p_code_added.rand, 
                                        showscale = False, 
                                        hovertext= gdf_p_code_added.NAME,
                                        marker = dict(opacity = 0.2),
                                        marker_line_width=.5))


        fig_m.update_layout(mapbox_style="open-street-map",
                        mapbox_center = {"lat": gdf_p_code_added['lat'].mean()+10, "lon": gdf_p_code_added['lon'].mean()},
                        height = 500,
                        width = 1000,
                        mapbox_zoom = 1.4,
                        autosize=True)

        return fig_m, 'Greater Vancouver A RDA (CSD, BC)'



    if type(clickData) == dict:
        # print(clickData['points'][0]['location'])

        clicked_code = str(clickData['points'][0]['location'])
        if len(clicked_code) == 2:
            
            region_codes = mapped_geo_code.query("Province_Code == " + f"'{clicked_code}'")['Region_Code'].unique()[1:]
            gdf_r_filtered = gdf_r.loc[region_codes, :]

            gdf_r_filtered["rand"] = np.random.randint(1, 100, len(gdf_r_filtered))
            
            # print(gdf_r_filtered.index)

            fig_mr = go.Figure()

            fig_mr.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_r_filtered.geometry.to_json()), 
                                            locations = gdf_r_filtered.index, 
                                            z = gdf_r_filtered.rand, 
                                            showscale = False, 
                                            hovertext= gdf_r_filtered.CDNAME,
                                            marker = dict(opacity = 0.2),
                                            marker_line_width=.5))


            fig_mr.update_layout(mapbox_style="open-street-map",
                            mapbox_center = {"lat": gdf_r_filtered['lat'].mean(), "lon": gdf_r_filtered['lon'].mean()},
                            height = 500,
                            width = 1000,
                            mapbox_zoom = 2.5,
                            autosize=True)
            # print('map is created')

            return fig_mr, 'Greater Vancouver A RDA (CSD, BC)'

        elif len(clicked_code) == 4:

            # print(clicked_code)

            # clicked_code = 5915
            subregion_codes = mapped_geo_code.query("Region_Code == " + f"'{clicked_code}'")['Geo_Code'].unique()[1:].astype(str)
            # print(subregion_codes)
            # gdf_sr_filtered = gdf_sr.loc[subregion_codes, :]
            gdf_sr_filtered = gpd.read_file(f'./sources/mapdata/subregion_data/{clicked_code}.shp')
            gdf_sr_filtered = gdf_sr_filtered.set_index('CSDUID')

            gdf_sr_filtered["rand"] = np.random.randint(1, 100, len(gdf_sr_filtered))

            # print(gdf_sr_filtered.index)

            fig_msr = go.Figure()

            fig_msr.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_sr_filtered.geometry.to_json()), 
                                            locations = gdf_sr_filtered.index, 
                                            z = gdf_sr_filtered.rand, 
                                            showscale = False, 
                                            hovertext= gdf_sr_filtered.CSDNAME,
                                            marker = dict(opacity = 0.2),
                                            marker_line_width=.5))

            max_bound = max(abs((gdf_sr_filtered['lat'].max() - gdf_sr_filtered['lat'].min())), 
                            abs((gdf_sr_filtered['lon'].max() - gdf_sr_filtered['lon'].min()))) * 111

            zoom = 11.5 - np.log(max_bound)
            # print(zoom)

            fig_msr.update_layout(mapbox_style="open-street-map",
                            mapbox_center = {"lat": gdf_sr_filtered['lat'].mean(), "lon": gdf_sr_filtered['lon'].mean()},
                            height = 500,
                            width = 1000,
                            mapbox_zoom = 11.5 - np.log(max_bound),
                            autosize=True)

            # print('map is created')

            return fig_msr, 'Greater Vancouver A RDA (CSD, BC)'

        elif len(clicked_code) > 4:

            # print(clicked_code)

            clicked_code_region = clicked_code[:4]


            subregion_codes = mapped_geo_code.query("Region_Code == " + f"'{clicked_code_region}'")['Geo_Code'].unique()[1:].astype(str)
            # print(subregion_codes)
            # gdf_sr_filtered = gdf_sr.loc[subregion_codes, :]
            gdf_sr_filtered = gpd.read_file(f'./sources/mapdata/subregion_data/{clicked_code_region}.shp')
            gdf_sr_filtered = gdf_sr_filtered.set_index('CSDUID')


            gdf_sr_filtered["rand"] = np.random.randint(1, 100, len(gdf_sr_filtered))

            # print(gdf_sr_filtered.index)

            fig_msr = go.Figure()

            fig_msr.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_sr_filtered.geometry.to_json()), 
                                            locations = gdf_sr_filtered.index, 
                                            z = gdf_sr_filtered.rand, 
                                            showscale = False, 
                                            hovertext= gdf_sr_filtered.CSDNAME,
                                            marker = dict(opacity = 0.2),
                                            marker_line_width=.5))

            max_bound = max(abs((gdf_sr_filtered['lat'].max() - gdf_sr_filtered['lat'].min())), 
                            abs((gdf_sr_filtered['lon'].max() - gdf_sr_filtered['lon'].min()))) * 111

            zoom = 11.5 - np.log(max_bound)
            # print(zoom)

            fig_msr.update_layout(mapbox_style="open-street-map",
                            mapbox_center = {"lat": gdf_sr_filtered['lat'].mean(), "lon": gdf_sr_filtered['lon'].mean()},
                            height = 500,
                            width = 1000,
                            mapbox_zoom = 11.5 - np.log(max_bound),
                            autosize=True)

            # print('map is created')

            subregion_name = mapped_geo_code.query("Geo_Code == " + f"{clicked_code}")['Geography'].tolist()[0]
            # print(subregion_name)
            return fig_msr, subregion_name

    else:
        # print(btn1)
        # print(ctx.triggered_id)
        gdf_p_code_added["rand"] = np.random.randint(1, 100, len(gdf_p_code_added))

        fig_m = go.Figure()

        fig_m.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_p_code_added.geometry.to_json()), 
                                        locations = gdf_p_code_added.index, 
                                        z = gdf_p_code_added.rand, 
                                        showscale = False, 
                                        hovertext= gdf_p_code_added.NAME,
                                        marker = dict(opacity = 0.2),
                                        marker_line_width=.5))


        fig_m.update_layout(mapbox_style="open-street-map",
                        mapbox_center = {"lat": gdf_p_code_added.geometry.centroid.y.mean()+10, "lon": gdf_p_code_added.geometry.centroid.x.mean()},
                        height = 500,
                        width = 1000,
                        mapbox_zoom = 1.4,
                        autosize=True)

        return fig_m, 'Greater Vancouver A RDA (CSD, BC)'




# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('datatable-interactivity', 'columns'),
    Output('datatable-interactivity', 'data'),
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('datatable-interactivity', 'selected_columns'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_table1(geo, geo_c, selected_columns, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]


        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        portion_of_total_hh = []
        for x in x_base:
            portion_of_total_hh.append(round(joined_df_filtered[f'Percent of Total HH that are in {x}'].tolist()[0] * 100, 2))

        amhi_list = []
        for a in amhi_range:
            amhi_list.append(joined_df_filtered[a].tolist()[0])

        shelter_range = ['20% or under of AMHI.1', '21% to 50% of AMHI.1', '51% to 80% of AMHI.1', '81% to 120% of AMHI.1', '121% and more of AMHI.1']
        shelter_list = []
        for s in shelter_range:
            shelter_list.append(joined_df_filtered[s].tolist()[0])

        table = pd.DataFrame({'Area Median HH Income': income_ct, 'Portion of total HHs(%)': portion_of_total_hh , 'Annual Household Income': amhi_list, 'Affordable shelter cost (2015 CAD$)': shelter_list})

        col_list = []

        for i in table.columns:
            col_list.append({"name": [geo, i], "id": i})

        return col_list, table.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns]
        
    else:


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]


        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        portion_of_total_hh = []
        for x in x_base:
            portion_of_total_hh.append(round(joined_df_filtered[f'Percent of Total HH that are in {x}'].tolist()[0] * 100, 2))

        amhi_list = []
        for a in amhi_range:
            amhi_list.append(joined_df_filtered[a].tolist()[0])

        shelter_range = ['20% or under of AMHI.1', '21% to 50% of AMHI.1', '51% to 80% of AMHI.1', '81% to 120% of AMHI.1', '121% and more of AMHI.1']
        shelter_list = []
        for s in shelter_range:
            shelter_list.append(joined_df_filtered[s].tolist()[0])

        table = pd.DataFrame({'Area Median HH Income': income_ct, '% of Total HHs': portion_of_total_hh , 'Annual HH Income': amhi_list, 'Affordable Shelter Cost': shelter_list})
        table['% of Total HHs'] = table['% of Total HHs'].astype(str) + '%'

        col_list = []

        for i in table.columns:
            if i == 'Area Median HH Income':
                col_list.append({"name": ["Income Category", i], "id": i})
            else:
                col_list.append({"name": [geo, i], "id": i})

        # Comparison

        if geo_c == None:
            geo_c = geo

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo_c}"')

        portion_of_total_hh = []
        for x in x_base:
            portion_of_total_hh.append(round(joined_df_filtered[f'Percent of Total HH that are in {x}'].tolist()[0] * 100, 2))

        amhi_list = []
        for a in amhi_range:
            amhi_list.append(joined_df_filtered[a].tolist()[0])

        shelter_range = ['20% or under of AMHI.1', '21% to 50% of AMHI.1', '51% to 80% of AMHI.1', '81% to 120% of AMHI.1', '121% and more of AMHI.1']
        shelter_list = []
        for s in shelter_range:
            shelter_list.append(joined_df_filtered[s].tolist()[0])

        table_c = pd.DataFrame({'Area Median HH Income': income_ct, '% of Total HHs ': portion_of_total_hh , 'Annual HH Income ': amhi_list, 'Affordable Shelter Cost ': shelter_list})
        table_c['% of Total HHs '] = table_c['% of Total HHs '].astype(str) + '%'

        table_j = table.merge(table_c, how = 'left', on = 'Area Median HH Income')

        for i in table_c.columns[1:]:
            col_list.append({"name": [geo_c, i], "id": i})

        return col_list, table_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns]




# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure(geo, geo_c, btn1, btn2, btn3):

    # if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

    #     if geo == None and geo_c != None:
    #         geo = geo_c
    #     elif geo == None and geo_c == None:
    #         geo = 'Greater Vancouver A RDA (CSD, BC)'


    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]



        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

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

        plot_df = pd.DataFrame({'Income_Category': x_list, 'Percent HH': joined_df_filtered[columns].T.iloc[:,0].tolist()})

        colors = ['#0B4952', '#4A8F97', '#4a5b97', '#FAB88A', '#FFDD5D', ]
        #colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

        fig = go.Figure()
        for i, c in zip(plot_df['Income_Category'], colors):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent HH'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>'
            ))

        fig.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor='#f0faff', title = f'Percent HH By Income Category - {geo}', legend_title = "Income")
        fig.update_xaxes(range = [0, 1])
            
        return fig

    else:
        # if geo == None:
        #     geo = 'Greater Vancouver A RDA (CSD, BC)'

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]


        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        x_list = []

        i = 0
        for b, c in zip(x_base, x_columns):
            if i < 4:
                b = b.replace(" Income", "")
                x = b + " ($" + str(joined_df_filtered[c].tolist()[0]) + ")"
                x_list.append(x)
            else:
                b = b.replace(" Income", "")
                x = b + " (>$" + str(joined_df_filtered[c].tolist()[0]) + ")"
                x_list.append(x)
            i += 1

        plot_df = pd.DataFrame({'Income_Category': x_list, 'Percent HH': joined_df_filtered[columns].T.iloc[:,0].tolist()})

        colors = ['#0B4952', '#4A8F97', '#4a5b97', '#FAB88A', '#FFDD5D', ]
        #colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_xaxes=True)

        n = 0
        for i, c, b in zip(plot_df['Income_Category'], colors, x_base):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent HH'],
                name = b.replace(" Income", ""),
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}'
            ), row = 1, col = 1)
            n += 1

        # Comparison plot

        joined_df_filtered_c = joined_df.query('Geography == '+ f'"{geo_c}"')

        x_list = []

        i = 0
        for b, c in zip(x_base, x_columns):
            if i < 4:
                b = b.replace(" Income", "")
                x = b + " ($" + str(joined_df_filtered_c[c].tolist()[0]) + ")"
                x_list.append(x)
            else:
                b = b.replace(" Income", "")
                x = b + " (>$" + str(joined_df_filtered_c[c].tolist()[0]) + ")"
                x_list.append(x)
            i += 1

        plot_df_c = pd.DataFrame({'Income_Category': x_list, 'Percent HH': joined_df_filtered_c[columns].T.iloc[:,0].tolist()})

        colors = ['#0B4952', '#4A8F97', '#4a5b97', '#FAB88A', '#FFDD5D', ]
        #colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

        n = 0
        for i, c, b in zip(plot_df_c['Income_Category'], colors, x_base):
            plot_df_frag_c = plot_df_c.loc[plot_df_c['Income_Category'] == i, :]
            fig.add_trace(go.Bar(
                y = plot_df_frag_c['Income_Category'],
                x = plot_df_frag_c['Percent HH'],
                name = b.replace(" Income", ""),
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}',
                showlegend = False
            ), row = 1, col = 2)
            n += 1


        fig.update_layout(title = f'Percent HH By Income Category', plot_bgcolor='#f0faff', legend_title = "Income", legend = dict(font = dict(size = 9)))
        fig.update_yaxes(tickfont = dict(size = 9.5), autorange = "reversed")
        fig.update_xaxes(range = [0, 1])

        return fig



# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph2', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure2(geo, geo_c, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

    # if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

    #     if geo == None and geo_c != None:
    #         geo = geo_c
    #     elif geo == None and geo_c == None:
    #         geo = 'Greater Vancouver A RDA (CSD, BC)'

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

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

        income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

        h_hold_value = []
        hh_p_num_list_full = []

        for h in hh_p_num_list:
            for i in income_lv_list:
                column = f'Per HH with income {i} of AMHI in core housing need that are {h} person HH'
                #print(geo, joined_df_filtered[column])
                h_hold_value.append(joined_df_filtered[column].tolist()[0])
                hh_p_num_list_full.append(h)

        plot_df = pd.DataFrame({'HH_Size': hh_p_num_list_full, 'Income_Category': x_list * 5, 'Percent': h_hold_value})

        # colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']
        colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

        fig2 = go.Figure()

        for h, c in zip(plot_df['HH_Size'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
            fig2.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'HH Size: {h} - ' + '%{x: .2%}<extra></extra>',
            ))
            
        fig2.update_layout(legend_traceorder = 'normal', yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = f'Percent HH By Income Category and AMHI - {geo}', legend_title = "Household Size")

        return fig2

    else:

        # if geo == None:
        #     geo = 'Greater Vancouver A RDA (CSD, BC)'

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]


        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        x_list = []

        i = 0
        for b, c in zip(x_base, x_columns):
            if i < 4:
                b = b.replace(" Income", "")
                x = b + " ($" + str(joined_df_filtered[c].tolist()[0]) + ")"
                x_list.append(x)
            else:
                b = b.replace(" Income", "")
                x = b + " (>$" + str(joined_df_filtered[c].tolist()[0]) + ")"
                x_list.append(x)
            i += 1

        income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

        h_hold_value = []
        hh_p_num_list_full = []

        for h in hh_p_num_list:
            for i in income_lv_list:
                column = f'Per HH with income {i} of AMHI in core housing need that are {h} person HH'
                #print(geo, joined_df_filtered[column])
                h_hold_value.append(joined_df_filtered[column].tolist()[0])
                hh_p_num_list_full.append(h)

        plot_df = pd.DataFrame({'HH_Size': hh_p_num_list_full, 'Income_Category': x_list * 5, 'Percent': h_hold_value})

        # colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']
        colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_xaxes=True)

        n = 0
        for h, c in zip(plot_df['HH_Size'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
            fig2.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'HH Size: {h} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}'
            ), row = 1, col = 1)
            n += 1
            
        # Comparison plot

        joined_df_filtered_c = joined_df.query('Geography == '+ f'"{geo_c}"')

        x_list = []

        i = 0
        for b, c in zip(x_base, x_columns):
            if i < 4:
                b = b.replace(" Income", "")
                x = b + " ($" + str(joined_df_filtered_c[c].tolist()[0]) + ")"
                x_list.append(x)
            else:
                b = b.replace(" Income", "")
                x = b + " (>$" + str(joined_df_filtered_c[c].tolist()[0]) + ")"
                x_list.append(x)
            i += 1

        income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

        h_hold_value = []
        hh_p_num_list_full = []

        for h in hh_p_num_list:
            for i in income_lv_list:
                column = f'Per HH with income {i} of AMHI in core housing need that are {h} person HH'
                #print(geo, joined_df_filtered[column])
                h_hold_value.append(joined_df_filtered_c[column].tolist()[0])
                hh_p_num_list_full.append(h)

        plot_df_c = pd.DataFrame({'HH_Size': hh_p_num_list_full, 'Income_Category': x_list * 5, 'Percent': h_hold_value})

        # colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']
        colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

        n = 0
        for h, c in zip(plot_df_c['HH_Size'].unique(), colors):
            plot_df_frag_c = plot_df_c.loc[plot_df_c['HH_Size'] == h, :]
            fig2.add_trace(go.Bar(
                y = plot_df_frag_c['Income_Category'],
                x = plot_df_frag_c['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'HH Size: {h} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}',
                showlegend = False
            ), row = 1, col = 2)
            n += 1

        fig2.update_layout(legend_traceorder = 'normal', barmode='stack', plot_bgcolor='#f0faff', title = f'Percent HH By Income Category and AMHI', legend_title = "Household Size", legend = dict(font = dict(size = 9)))
        fig2.update_yaxes(tickfont = dict(size = 9.5), autorange = "reversed")

        return fig2



# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('datatable2-interactivity', 'columns'),
    Output('datatable2-interactivity', 'data'),
    Output('datatable2-interactivity', 'style_data_conditional'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('datatable2-interactivity', 'selected_columns'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_table2(geo, geo_c, selected_columns, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]


    # if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

    #     if geo == None and geo_c != None:
    #         geo = geo_c
    #     elif geo == None and geo_c == None:
    #         geo = 'Greater Vancouver A RDA (CSD, BC)'

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        table2 = pd.DataFrame({'Income Group': income_ct})

        h_hold_value = []
        hh_p_num_list = [1,2,3,4,'5 or more']
        income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

        for h in hh_p_num_list:
            h_hold_value = []
            if h == 1:
                h2 = '1 person'
            elif h == '5 or more':
                h2 = '5 or more persons household'
            else:
                h2 = f'{str(h)} persons'
            for i in income_lv_list:
                if i == '20% or under':
                    column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of area median household income (AMHI)-Households in core housing need'
                    h_hold_value.append(joined_df_filtered[column].tolist()[0])

                else:
                    column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of AMHI-Households in core housing need'
                    h_hold_value.append(joined_df_filtered[column].tolist()[0])
            if h == 1:        
                table2[f'{h} P HH'] = h_hold_value
            elif h == '5 or more':
                table2[f'5 >= P HH'] = h_hold_value
            else:
                table2[f'{h} P HH'] = h_hold_value

        table2['Total'] = table2.sum(axis = 1)

        col_list = []

        for i in table2.columns:
            col_list.append({"name": [geo, i], "id": i})

        return col_list, table2.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns]


    else:
        # if geo == None:
        #     geo = geo_c

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]


        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        table2 = pd.DataFrame({'Area Median HH Income': income_ct})

        h_hold_value = []
        hh_p_num_list = [1,2,3,4,'5 or more']
        income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

        for h in hh_p_num_list:
            h_hold_value = []
            if h == 1:
                h2 = '1 person'
            elif h == '5 or more':
                h2 = '5 or more persons household'
            else:
                h2 = f'{str(h)} persons'
            for i in income_lv_list:
                if i == '20% or under':
                    column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of area median household income (AMHI)-Households in core housing need'
                    h_hold_value.append(joined_df_filtered[column].tolist()[0])

                else:
                    column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of AMHI-Households in core housing need'
                    h_hold_value.append(joined_df_filtered[column].tolist()[0])
            if h == 1:        
                table2[f'{h} P HH'] = h_hold_value
            elif h == '5 or more':
                table2[f'5 >= P HH'] = h_hold_value
            else:
                table2[f'{h} P HH'] = h_hold_value

        table2['Total'] = table2.sum(axis = 1)

        col_list = []

        for i in table2.columns:
            if i == 'Area Median HH Income':
                col_list.append({"name": ["Income Category", i], "id": i})
            else:
                col_list.append({"name": [geo, i], "id": i})

        # Comparison Table

        if geo == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'

        joined_df_filtered_c = joined_df.query('Geography == '+ f'"{geo_c}"')

        table2_c = pd.DataFrame({'Area Median HH Income': income_ct})

        h_hold_value = []
        hh_p_num_list = [1,2,3,4,'5 or more']
        income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

        for h in hh_p_num_list:
            h_hold_value = []
            if h == 1:
                h2 = '1 person'
            elif h == '5 or more':
                h2 = '5 or more persons household'
            else:
                h2 = f'{str(h)} persons'
            for i in income_lv_list:
                if i == '20% or under':
                    column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of area median household income (AMHI)-Households in core housing need'
                    h_hold_value.append(joined_df_filtered_c[column].tolist()[0])

                else:
                    column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of AMHI-Households in core housing need'
                    h_hold_value.append(joined_df_filtered_c[column].tolist()[0])
            if h == 1:        
                table2_c[f'{h} P HH '] = h_hold_value
            elif h == '5 or more':
                table2_c[f'5 >= P HH '] = h_hold_value
            else:
                table2_c[f'{h} P HH '] = h_hold_value

        table2_c['Total '] = table2.sum(axis = 1)

        for i in table2_c.columns[1:]:
            if i == 'Area Median HH Income':
                col_list.append({"name": ["Income Category", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], "id": i})

        table2_j = table2.merge(table2_c, how = 'left', on = 'Area Median HH Income')

        return col_list, table2_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns]





# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph5', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure5(geo, geo_c, btn1, btn2, btn3):



    hh_category_dict = {
                'Percent Women-led HH in core housing need' : 'Women-led HH', 
                'Percent Single Mother led HH in core housing need' : 'Single mother-led HH', 
                'Percent Indigenous HH in core housing need' : 'Indigenous HH', 
                'Percent Visible minority HH in core housing need' : 'Visible minority HH', 
                'Percent Black-led HH in core housing need' : 'Black-led HH', 
                'Percent New migrant-led HH in core housing need' : 'New migrant-led HH', 
                'Percent Refugee claimant-led HH in core housing need' : 'Refugee claimant-led HH', 
                'Percent HH head under 25 in core housing need' : 'HH head under 25', 
                'Percent HH head over 65 in core housing need' : 'HH head over 65', 
                'Percent HH head over 85 in core housing need' : 'HH head over 85', 
                'Percent HH with physical act. limit. in core housing need' : 'HH with physical activity limitation', 
                'Percent HH with cognitive, mental, or addictions activity limitation in core housing need' : 'HH with cognitive, mental, or addictions activity limitation', 
                'Percent HH in core housing need' : 'Community (all HH)'
            }

    columns = hh_category_dict.keys()
    hh_categories = list(hh_category_dict.values())

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]


    # if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

    #     if geo == None and geo_c != None:
    #         geo = geo_c
    #     elif geo == None and geo_c == None:
    #         geo = 'Greater Vancouver A RDA (CSD, BC)'

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        percent_hh = [joined_df_filtered[c].tolist()[0] for c in columns]

        plot_df = pd.DataFrame({'HH_Category': hh_categories, 'Percent_HH': percent_hh})
        plot_df['Percent_HH'] = plot_df['Percent_HH'].fillna(0)

        # colors = ['#fff194','#4A8F97', '#210b52', '#0B4952', '#FFDD5D', '#158232', '#4a5b97', '#6ed0db', '#bfd5ff', '#ff8d3d', '#166370', '#FAB88A',  '#ffe28f']
        color_dict = {}

        for h in plot_df['HH_Category'].unique():
            
            if plot_df['Percent_HH'].max() == 0:
                color_dict[h] = '#bfd5ff'
            else:
                if h == plot_df.loc[plot_df['Percent_HH'] == plot_df['Percent_HH'].max(), 'HH_Category'].tolist()[0]:
                        color_dict[h] = '#4a5b97'
                elif h == 'Community (all HH)':
                    color_dict[h] = '#158232'
                else:
                    color_dict[h] = '#bfd5ff'

        fig5 = go.Figure()
        for i in hh_categories:
            plot_df_frag = plot_df.loc[plot_df['HH_Category'] == i, :]
            fig5.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent_HH'],
                name = i,
                marker_color = color_dict[i],
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                
            ))
        fig5.update_layout(yaxis=dict(autorange="reversed"), showlegend = False, plot_bgcolor='#f0faff', title = f'Percentage of HHs in Core Housing Need by Priority Population - {geo}', legend_title = "HH Category")

        return fig5

    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]


        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        percent_hh = [joined_df_filtered[c].tolist()[0] for c in columns]

        plot_df = pd.DataFrame({'HH_Category': hh_categories, 'Percent_HH': percent_hh})
        plot_df['Percent_HH'] = plot_df['Percent_HH'].fillna(0)

        # colors = ['#fff194','#4A8F97', '#210b52', '#0B4952', '#FFDD5D', '#158232', '#4a5b97', '#6ed0db', '#bfd5ff', '#ff8d3d', '#166370', '#FAB88A',  '#ffe28f']
        color_dict = {}

        for h in plot_df['HH_Category'].unique():
            
            if plot_df['Percent_HH'].max() == 0:
                color_dict[h] = '#bfd5ff'
            else:
                if h == plot_df.loc[plot_df['Percent_HH'] == plot_df['Percent_HH'].max(), 'HH_Category'].tolist()[0]:
                        color_dict[h] = '#4a5b97'
                elif h == 'Community (all HH)':
                    color_dict[h] = '#158232'
                else:
                    color_dict[h] = '#bfd5ff'

        fig5 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i in hh_categories:
            plot_df_frag = plot_df.loc[plot_df['HH_Category'] == i, :]
            fig5.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent_HH'],
                name = i,
                marker_color = color_dict[i],
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                
            ),row = 1, col = 1)


        # Comparison Plot

        joined_df_filtered_c = joined_df.query('Geography == '+ f'"{geo_c}"')

        percent_hh = [joined_df_filtered_c[c].tolist()[0] for c in columns]

        plot_df_c = pd.DataFrame({'HH_Category': hh_categories, 'Percent_HH': percent_hh})
        plot_df_c['Percent_HH'] = plot_df_c['Percent_HH'].fillna(0)

        # colors = ['#fff194','#4A8F97', '#210b52', '#0B4952', '#FFDD5D', '#158232', '#4a5b97', '#6ed0db', '#bfd5ff', '#ff8d3d', '#166370', '#FAB88A',  '#ffe28f']
        color_dict = {}

        for h in plot_df_c['HH_Category'].unique():
            
            if plot_df_c['Percent_HH'].max() == 0:
                color_dict[h] = '#bfd5ff'
            else:
                if h == plot_df_c.loc[plot_df_c['Percent_HH'] == plot_df_c['Percent_HH'].max(), 'HH_Category'].tolist()[0]:
                        color_dict[h] = '#4a5b97'
                elif h == 'Community (all HH)':
                    color_dict[h] = '#158232'
                else:
                    color_dict[h] = '#bfd5ff'

        for i in hh_categories:
            plot_df_frag_c = plot_df_c.loc[plot_df_c['HH_Category'] == i, :]
            fig5.add_trace(go.Bar(
                y = plot_df_frag_c['HH_Category'],
                x = plot_df_frag_c['Percent_HH'],
                name = i,
                marker_color = color_dict[i],
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                
            ),row = 1, col = 2)
        fig5.update_layout(yaxis=dict(autorange="reversed"), showlegend = False, plot_bgcolor='#f0faff', title = f'Percentage of HHs in Core Housing Need by Priority Population', legend_title = "HH Category")

        return fig5




# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph6', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure6(geo, geo_c, btn1, btn2, btn3):

    hh_category_dict2 = {
                        'Percent of Women-led HH in core housing need' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing need' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing need' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing need' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing need' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing need' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing need' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing need' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing need' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing need' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing need' : 'HH with physical activity limitation', 
                        'Percent of HH with with cognitive, mental, or addictions activity limitation in core housing need' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    hh_category_dict3 = {
                        'Percent of Women-led HH core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation', 
                        }

    hh_category_dict4 = {
                        'Percent of Women-led HH in core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    columns2 = hh_category_dict2.keys()
    columns3 = hh_category_dict3.keys()
    columns4 = hh_category_dict4.keys()

    income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]


    # if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

    #     if geo == None and geo_c != None:
    #         geo = geo_c
    #     elif geo == None and geo_c == None:
    #         geo = 'Greater Vancouver A RDA (CSD, BC)'

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        income_col = []
        percent_col = []
        hh_cat_col = []

        for c, c2, c3 in zip(columns2, columns3, columns4):
            for i in income_lv_list:
                if i == '20% or under':
                    p_hh = joined_df_filtered[f'{c} with income {i} of the AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict2[c])
                elif i == '21% to 50%':
                    p_hh = joined_df_filtered[f'{c2} with income {i} of AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict3[c2])
                else:
                    p_hh = joined_df_filtered[f'{c3} with income {i} of AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                    
                income_col.append(i)
                percent_col.append(p_hh)

        plot_df = pd.DataFrame({'Income_Category': income_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

        fig6 = go.Figure()

        colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']
        # colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

        for i, c in zip(plot_df['Income_Category'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig6.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'Income Level: {i} - ' + '%{x: .2%}<extra></extra>',
            ))
            
        fig6.update_layout(legend_traceorder="normal", yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = f'Percentage of HHs in Core Housing Need by Priority Population and Income - {geo}', legend_title = "Income Category")

        return fig6

    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        income_col = []
        percent_col = []
        hh_cat_col = []

        for c, c2, c3 in zip(columns2, columns3, columns4):
            for i in income_lv_list:
                if i == '20% or under':
                    p_hh = joined_df_filtered[f'{c} with income {i} of the AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict2[c])
                elif i == '21% to 50%':
                    p_hh = joined_df_filtered[f'{c2} with income {i} of AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict3[c2])
                else:
                    p_hh = joined_df_filtered[f'{c3} with income {i} of AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                    
                income_col.append(i)
                percent_col.append(p_hh)

        plot_df = pd.DataFrame({'Income_Category': income_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

        fig6 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']
        # colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

        n = 0
        for i, c in zip(plot_df['Income_Category'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig6.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'Income Level: {i} - ' + '%{x: .2%}<extra></extra>',
                legendgroup= f'{n}'
            ), row = 1, col = 1)
            n += 1
            

        # Comparison plot

        joined_df_filtered_c = joined_df.query('Geography == '+ f'"{geo_c}"')

        income_col = []
        percent_col = []
        hh_cat_col = []

        for c, c2, c3 in zip(columns2, columns3, columns4):
            for i in income_lv_list:
                if i == '20% or under':
                    p_hh = joined_df_filtered_c[f'{c} with income {i} of the AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict2[c])
                elif i == '21% to 50%':
                    p_hh = joined_df_filtered_c[f'{c2} with income {i} of AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict3[c2])
                else:
                    p_hh = joined_df_filtered_c[f'{c3} with income {i} of AMHI'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                    
                income_col.append(i)
                percent_col.append(p_hh)

        plot_df_c = pd.DataFrame({'Income_Category': income_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

        colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']
        # colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

        n = 0
        for i, c in zip(plot_df_c['Income_Category'].unique(), colors):
            plot_df_frag_c = plot_df_c.loc[plot_df_c['Income_Category'] == i, :]
            fig6.add_trace(go.Bar(
                y = plot_df_frag_c['HH_Category'],
                x = plot_df_frag_c['Percent'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate = '%{y}, ' + f'Income Level: {i} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}',
                showlegend = False
            ), row = 1, col = 2)
            n += 1
            
        fig6.update_layout(legend_traceorder="normal", yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = f'Percentage of HHs in Core Housing Need by Priority Population and Income', legend_title = "Income Category")

        return fig6





# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph7', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure7(geo, geo_c, btn1, btn2, btn3):

    hh_category_dict2 = {
                        'Percent of Women-led HH in core housing need' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing need' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing need' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing need' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing need' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing need' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing need' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing need' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing need' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing need' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing need' : 'HH with physical activity limitation', 
                        'Percent of HH with with cognitive, mental, or addictions activity limitation in core housing need' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    hh_category_dict3 = {
                        'Percent of Women-led HH core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation', 
                        }

    hh_category_dict4 = {
                        'Percent of Women-led HH in core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    columns2 = hh_category_dict2.keys()
    columns3 = hh_category_dict3.keys()
    columns4 = hh_category_dict4.keys()

    # if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

    #     if geo == None and geo_c != None:
    #         geo = geo_c
    #     elif geo == None and geo_c == None:
    #         geo = 'Greater Vancouver A RDA (CSD, BC)'

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        hh_size_col = []
        percent_col = []
        hh_cat_col = []

        hh_size = ['1 person', '2 people', '3 people', '4 people', '5 or more people']

        for c, c2, c3 in zip(columns2, columns3, columns4):
            for h in hh_size:
                if h == '1 person':
                    p_hh = joined_df_filtered[f'{c} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict2[c])
                elif h == '2 people':
                    p_hh = joined_df_filtered[f'{c2} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict3[c2])
                elif h == '5 or more people':
                    p_hh = joined_df_filtered[f'{c3}with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                else:
                    p_hh = joined_df_filtered[f'{c3} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                    
                hh_size_col.append(h)
                percent_col.append(p_hh)

        plot_df = pd.DataFrame({'HH_Size': hh_size_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

        fig7 = go.Figure()

        # colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']
        colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

        for h, c in zip(plot_df['HH_Size'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
            fig7.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'Income Level: {h} - ' + '%{x: .2%}<extra></extra>',
            ))
                
        fig7.update_layout(legend_traceorder="normal", yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = f'Percentage of Households (HHs) in Core Housing Need by Priority Population and HH Size - {geo}', legend_title = "HH Size")

        return fig7

    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

        hh_size_col = []
        percent_col = []
        hh_cat_col = []

        hh_size = ['1 person', '2 people', '3 people', '4 people', '5 or more people']

        for c, c2, c3 in zip(columns2, columns3, columns4):
            for h in hh_size:
                if h == '1 person':
                    p_hh = joined_df_filtered[f'{c} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict2[c])
                elif h == '2 people':
                    p_hh = joined_df_filtered[f'{c2} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict3[c2])
                elif h == '5 or more people':
                    p_hh = joined_df_filtered[f'{c3}with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                else:
                    p_hh = joined_df_filtered[f'{c3} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                    
                hh_size_col.append(h)
                percent_col.append(p_hh)

        plot_df = pd.DataFrame({'HH_Size': hh_size_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

        fig7 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        # colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']
        colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

        n = 0
        for h, c in zip(plot_df['HH_Size'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
            fig7.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'Income Level: {h} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}'
            ),row = 1, col = 1)
            n += 1

        # Comparison Plot

        joined_df_filtered_c = joined_df.query('Geography == '+ f'"{geo_c}"')

        hh_size_col = []
        percent_col = []
        hh_cat_col = []

        hh_size = ['1 person', '2 people', '3 people', '4 people', '5 or more people']

        for c, c2, c3 in zip(columns2, columns3, columns4):
            for h in hh_size:
                if h == '1 person':
                    p_hh = joined_df_filtered_c[f'{c} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict2[c])
                elif h == '2 people':
                    p_hh = joined_df_filtered_c[f'{c2} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict3[c2])
                elif h == '5 or more people':
                    p_hh = joined_df_filtered_c[f'{c3}with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                else:
                    p_hh = joined_df_filtered_c[f'{c3} with a HH size of {h}'].tolist()[0]
                    hh_cat_col.append(hh_category_dict4[c3])
                    
                hh_size_col.append(h)
                percent_col.append(p_hh)

        plot_df_c = pd.DataFrame({'HH_Size': hh_size_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

        # colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']
        colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

        n = 0
        for h, c in zip(plot_df_c['HH_Size'].unique(), colors):
            plot_df_frag_c = plot_df_c.loc[plot_df_c['HH_Size'] == h, :]
            fig7.add_trace(go.Bar(
                y = plot_df_frag_c['HH_Category'],
                x = plot_df_frag_c['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'Income Level: {h} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}',
                showlegend = False
            ),row = 1, col = 2)
            n += 1
                
        fig7.update_layout(legend_traceorder="normal", yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = f'Percentage of Households (HHs) in Core Housing Need by Priority Population and HH Size - {geo}', legend_title = "HH Size")

        return fig7





# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('datatable3-interactivity', 'columns'),
    Output('datatable3-interactivity', 'data'),
    Output('datatable3-interactivity', 'style_data_conditional'),
    Output('datatable4-interactivity', 'columns'),
    Output('datatable4-interactivity', 'data'),
    Output('datatable4-interactivity', 'style_data_conditional'),
    Output('datatable5-interactivity', 'columns'),
    Output('datatable5-interactivity', 'data'),
    Output('datatable5-interactivity', 'style_data_conditional'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('datatable3-interactivity', 'selected_columns'),
    Input('datatable4-interactivity', 'selected_columns'),
    Input('datatable5-interactivity', 'selected_columns'),
    # Input('to-geography-1', 'n_clicks'),
    # Input('to-region-1', 'n_clicks'),
    # Input('to-province-1', 'n_clicks')
)
def update_table3(geo, geo_c, selected_columns, selected_columns2, selected_columns3):#, btn1, btn2, btn3):

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

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'

        # if "to-geography-1" == ctx.triggered_id:
        #     geo = geo
        # elif "to-region-1" == ctx.triggered_id:
        #     geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        # elif "to-province-1" == ctx.triggered_id:
        #     geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        df_csd_proj_merged_filtered = df_csd_proj_merged.loc[df_csd_proj_merged['Geography'] == geo,:]
        geo_region = int(mapped_geo_code.loc[mapped_geo_code['Geography'] == geo]['Region_Code'].values[0])
        df_cd_proj_merged_filtered = df_cd_proj_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region,:]
        df_cd_grow_merged_filtered = df_cd_grow_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region,:]

        geo_region_name = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]

        income_l = []
        pp_l = []
        result_csd_l = []
        result_cd_l = []
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
                result_cd_l.append(df_cd_proj_merged_filtered[col_format_p].tolist()[0])
                result_t_l.append(df_csd_proj_merged_filtered[col_format_r].tolist()[0])
                result_g_l.append(df_cd_grow_merged_filtered[col_format_g].tolist()[0])
                
        table3 = pd.DataFrame({'Income_Category': income_l, 'HH_Category': pp_l, 'CSD_Projection': result_csd_l, 'CD_Projection': result_cd_l, 'CSD_Total': result_t_l, 'Growth': result_g_l})
        table3 = table3.fillna(0)
        table3['CSD_Total'] = table3['CSD_Total'].astype(float)
        table3['Growth'] = table3['Growth'].astype(float)
        table3['CSD_Projection'] = np.round(table3['CSD_Projection'].astype(float), 2)
        table3['CD_Projection'] = np.round(table3['CD_Projection'].astype(float), 2)
        table3['Projection'] = np.round((table3['CSD_Total'] * table3['Growth']) + table3['CSD_Total'], 2)

        table3_csd = table3.pivot_table(values='CSD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_csd = table3_csd.reset_index()

        table3_cd_r = table3.pivot_table(values='Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_cd_r = table3_cd_r.reset_index()

        table3_cd = table3.pivot_table(values='CD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_cd = table3_cd.reset_index()

        col_list_csd = []

        for i in table3_csd.columns:
            col_list_csd.append({"name": [geo, i], "id": i})

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            col_list_cd_r.append({"name": [geo_region_name, i], "id": i})

        col_list_cd = []

        for i in table3_cd.columns:
            col_list_cd.append({"name": [geo_region_name, i], "id": i})

        return col_list_csd, \
                table3_csd.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                col_list_cd_r, \
                table3_cd_r.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                col_list_cd, \
                table3_cd.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns3]


    else:
        # if geo == None:
        #     geo = geo_c

        # if "to-geography-1" == ctx.triggered_id:
        #     geo = geo
        #     geo_c = geo_c
        # elif "to-region-1" == ctx.triggered_id:
        #     geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        #     geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        # elif "to-province-1" == ctx.triggered_id:
        #     geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
        #     geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        df_csd_proj_merged_filtered = df_csd_proj_merged.loc[df_csd_proj_merged['Geography'] == geo,:]
        geo_region = int(mapped_geo_code.loc[mapped_geo_code['Geography'] == geo]['Region_Code'].values[0])
        df_cd_proj_merged_filtered = df_cd_proj_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region,:]
        df_cd_grow_merged_filtered = df_cd_grow_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region,:]

        geo_region_name = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]

        income_l = []
        pp_l = []
        result_csd_l = []
        result_cd_l = []
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
                result_cd_l.append(df_cd_proj_merged_filtered[col_format_p].tolist()[0])
                result_t_l.append(df_csd_proj_merged_filtered[col_format_r].tolist()[0])
                result_g_l.append(df_cd_grow_merged_filtered[col_format_g].tolist()[0])
                
        table3 = pd.DataFrame({'Income_Category': income_l, 'HH_Category': pp_l, 'CSD_Projection': result_csd_l, 'CD_Projection': result_cd_l, 'CSD_Total': result_t_l, 'Growth': result_g_l})
        table3 = table3.fillna(0)
        table3['CSD_Total'] = table3['CSD_Total'].astype(float)
        table3['Growth'] = table3['Growth'].astype(float)
        table3['CSD_Projection'] = np.round(table3['CSD_Projection'].astype(float), 2)
        table3['CD_Projection'] = np.round(table3['CD_Projection'].astype(float), 2)
        table3['Projection'] = np.round((table3['CSD_Total'] * table3['Growth']) + table3['CSD_Total'], 2)

        table3_csd = table3.pivot_table(values='CSD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_csd = table3_csd.reset_index()

        table3_cd_r = table3.pivot_table(values='Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_cd_r = table3_cd_r.reset_index()

        table3_cd = table3.pivot_table(values='CD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_cd = table3_cd.reset_index()

        col_list_csd = []

        for i in table3_csd.columns:
            col_list_csd.append({"name": [geo, i], "id": i})

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            col_list_cd_r.append({"name": [geo_region_name, i], "id": i})

        col_list_cd = []

        for i in table3_cd.columns:
            col_list_cd.append({"name": [geo_region_name, i], "id": i})


        # Comparison Table

        df_csd_proj_merged_filtered_c = df_csd_proj_merged.loc[df_csd_proj_merged['Geography'] == geo_c,:]
        geo_region_c = int(mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c]['Region_Code'].values[0])
        df_cd_proj_merged_filtered_c = df_cd_proj_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region_c,:]
        df_cd_grow_merged_filtered_c = df_cd_grow_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region_c,:]

        geo_region_name_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]

        income_l = []
        pp_l = []
        result_csd_l = []
        result_cd_l = []
        result_g_l = []
        result_t_l = []

        for i, i_r in zip(income_col_list, income_col_list_r):
            for p, p_r in zip(pp_list, pp_list_r):
                col_format_r = f'TotalPrivatehouseholdsbyhouseholdtypeincludingcensusfamilystructure - Householdswithincome{i_r} - {p_r} - 2016'
                col_format_p = f'2026 Projected {p} HH with income {i}'
                col_format_g = f'2026 Projected Growth {p} HH with income {i}'
                income_l.append(i)
                pp_l.append(p)
                result_csd_l.append(df_csd_proj_merged_filtered_c[col_format_p].tolist()[0])
                result_cd_l.append(df_cd_proj_merged_filtered_c[col_format_p].tolist()[0])
                result_t_l.append(df_csd_proj_merged_filtered_c[col_format_r].tolist()[0])
                result_g_l.append(df_cd_grow_merged_filtered_c[col_format_g].tolist()[0])
                
        table3_c = pd.DataFrame({'Income_Category': income_l, 'HH_Category': pp_l, 'CSD_Projection': result_csd_l, 'CD_Projection': result_cd_l, 'CSD_Total': result_t_l, 'Growth': result_g_l})
        table3_c = table3_c.fillna(0)
        table3_c['CSD_Total'] = table3_c['CSD_Total'].astype(float)
        table3_c['Growth'] = table3_c['Growth'].astype(float)
        table3_c['CSD_Projection'] = np.round(table3_c['CSD_Projection'].astype(float), 2)
        table3_c['CD_Projection'] = np.round(table3_c['CD_Projection'].astype(float), 2)
        table3_c['Projection'] = np.round((table3_c['CSD_Total'] * table3_c['Growth']) + table3_c['CSD_Total'], 2)

        table3_csd_c = table3_c.pivot_table(values='CSD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_csd_c = table3_csd_c.reset_index()
        table3_csd_c = table3_csd_c.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })

        table3_cd_r_c = table3_c.pivot_table(values='Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_cd_r_c = table3_cd_r_c.reset_index()
        table3_cd_r_c = table3_cd_r_c.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })

        table3_cd_c = table3_c.pivot_table(values='CD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
        table3_cd_c = table3_cd_c.reset_index()
        table3_cd_c = table3_cd_c.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })

        for i in table3_csd_c.columns[1:]:
            col_list_csd.append({"name": [geo_c, i], "id": i})

        for i in table3_cd_r_c.columns[1:]:
            col_list_cd_r.append({"name": [geo_region_name_c, i], "id": i})

        for i in table3_cd_c.columns[1:]:
            col_list_cd.append({"name": [geo_region_name_c, i], "id": i})

        table3_csd_j = table3_csd.merge(table3_csd_c, how = 'left', on = 'Income_Category')
        table3_cd_j = table3_cd.merge(table3_cd_c, how = 'left', on = 'Income_Category')
        table3_cd_r_j = table3_cd_r.merge(table3_cd_r_c, how = 'left', on = 'Income_Category')
        
        # print(table3_cd_j)

        return col_list_csd, \
                table3_csd_j.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                col_list_cd_r, \
                table3_cd_r_j.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                col_list_cd, \
                table3_cd_j.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns3]


# Creating raw csv data file for download option

@app.callback(
    Output("ov7-download-text", "data"),
    Input("ov7-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov7(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# To run Dash surver

if __name__ == "__main__":
    app.run_server(debug=True)