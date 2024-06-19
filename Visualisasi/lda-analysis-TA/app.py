import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc 

app = Dash(__name__, use_pages=True,external_stylesheets=[dbc.themes.PULSE])
app.title = "Sentiment analysis of applications based on LDA topic modeling"

navbar = dbc.NavbarSimple(
    brand="📱 Sentiment Analysis with Topic Modeling - LDA analysis output",
    color="primary",
    dark=True,
    )

body_layout = dbc.Container([
    html.Div([
        dbc.Card(
            dbc.CardBody(
                dbc.Col([
                    html.A(dbc.Badge("Home", color="primary", className="p-3 mr-3"), href="/"),
                    html.A(dbc.Badge("Dataset I", color="primary", className="p-3 mr-3"), href="/d1"),
                    html.A(dbc.Badge("Dataset II", color="primary", className="p-3 mx-3"), href="/d2"),
                    html.A(dbc.Badge("Dataset III", color="primary", className="p-3 mr-3"), href="/d3"),
                    ], className='d-flex flex-row')
                ),
            className="my-3",
            ),
        ]),
    dash.page_container
    ])

app.layout = html.Div([
    navbar, body_layout
    ])

if __name__ == '__main__':
    app.run(debug=True)