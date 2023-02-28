# Import necessary libraries
from dash import html
import dash_bootstrap_components as dbc

# Define the navbar structure
def Navbar():

    layout = html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Select Area", href="/page1")),
                dbc.NavItem(dbc.NavLink("Main", href="/page2")),
                dbc.NavItem(dbc.NavLink("Projection", href="/page3")),
            ] ,
            brand="HART Dashboard",
            brand_href="/page1",
            color="dark",
            dark=True,
        ), 
    ])

    return layout