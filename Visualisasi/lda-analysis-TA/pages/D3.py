from flask import Flask, render_template
import dash
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
import json
import dash_cytoscape as cyto

server = Flask(__name__)

app = dash.Dash(external_stylesheets=[dbc.themes.PULSE], server = server)
app.title = "Sentiment analysis of applications based on LDA topic modeling (Social Reiews Apps)"

df = pd.read_csv("dataset/moba-dataset.csv")
# Prep data / fill NAs
df["topic_id"] = df["topic_id"].astype(str)
topic_ids = [str(i) for i in range(len(df["topic_id"].unique()))]

# Topic Modeling LDA
with open("dataset/topics_moba.json", "r") as f:
    lda_topics = json.load(f)
    topics_txt = [lda_topics[str(i)] for i in range(len(lda_topics))]
    topics_txt = [[j.split("*")[1].replace('"', "") for j in i] for i in topics_txt]
    topics_txt = ["; ".join(i) for i in topics_txt]

    col_swatch = px.colors.qualitative.Dark24
    def_stylesheet = [
    {
    "selector": "." + str(i),
    "style": {"background-color": col_swatch[i], "line-color": col_swatch[i]},
    }
    for i in range(len(df["topic_id"].unique()))
    ]

# Topic view
topics_html = list()
for topic_html in [
    html.Span([str(i) + ": " + topics_txt[i]], style={"color": col_swatch[i]})
    for i in range(len(topics_txt))
    ]:
    topics_html.append(topic_html)
    topics_html.append(html.Br())

# Coherence Model
coherence = pd.read_csv('dataset/lda_tuning_moba(1).csv')
coherence = coherence.sort_values(['Coherence'], ascending=[False])
coherence = coherence.head(10)

# t-SNE test based on perplexity
tSNE = pd.read_csv("dataset/moba-dataset.csv")
tSNE["topic_id"] = tSNE["topic_id"].astype(str)
tSNE1 = px.scatter(tSNE, x="x1", y="y1", color="topic_id", symbol="topic_id")
tSNE10 = px.scatter(tSNE, x="x10", y="y10", color="topic_id", symbol="topic_id")
tSNE25 = px.scatter(tSNE, x="x25", y="y25", color="topic_id", symbol="topic_id")
tSNE100 = px.scatter(tSNE, x="x100", y="y100", color="topic_id", symbol="topic_id")

# Sample data
apps = ['AoV', 'Mobilelegends', 'Netdragons']
categories = ['game', 'hero', 'kalah', 'hp', 'lag', 'tim', 'karakter', 'pakai']
sentiment = ['negative', 'neutral', 'positive']

data_app = {
    'AoV': {
        'negative': [52.7, 39.6, 33.3, 57.1, 53.8, 81.8, 23.1, 71.4],
        'neutral': [6.5, 4.2, 33.3, 14.3, 15.4, 4.5, 0.0, 0.0],
        'positive': [40.8, 56.3, 33.3, 28.6, 30.8, 13.6, 76.9, 28.6],
    },
    'Mobilelegends': {
        'negative': [74.7, 76.5, 100.0, 66.7, 61.5, 85.0, 33.3, 100.0],
        'neutral': [8.2, 3.9, 0.0, 33.3, 0.0, 2.5, 33.3, 0.0],
        'positive': [17.1, 19.6, 0.0, 0.0, 38.5, 12.5, 33.3, 0.0],
    },
    'Netdragons': {
        'negative': [43.8, 41.9, 33.3, 60.0, 50.0, 72.2, 25.0, 36.4],
        'neutral': [15.0, 9.7, 0.0, 0.0, 11.1, 5.6, 0.0, 45.5],
        'positive': [41.2, 48.4, 66.7, 40.0, 38.9, 22.2, 75.0, 18.2],
    },
}

data_topic = {
    'game': {
        'AoV': [52.7, 6.5, 40.8],
        'Mobilelegends': [74.7, 8.2, 17.1],
        'Netdragons': [43.8, 15.0, 41.2],
    },
    'hero': {
        'AoV': [39.6, 4.2, 56.3],
        'Mobilelegends': [76.5, 3.9, 19.6],
        'Netdragons': [41.9, 9.7, 48.4],
    },
    'kalah': {
        'AoV': [33.3, 33.3, 33.3],
        'Mobilelegends': [100.0, 0.0, 0.0],
        'Netdragons': [33.3, 0.0, 66.7],
    },
    'hp': {
        'AoV': [57.1, 14.3, 28.6],
        'Mobilelegends': [66.7, 33.3, 0.0],
        'Netdragons': [60.0, 0.0, 40.0],
    },
    'lag': {
        'AoV': [53.8, 15.4, 30.8],
        'Mobilelegends': [61.5, 0.0, 38.5],
        'Netdragons': [50.0, 11.1, 38.9],
    },
    'tim': {
        'AoV': [81.8, 4.5, 13.6],
        'Mobilelegends': [85.0, 2.5, 12.5],
        'Netdragons': [72.2, 5.6, 22.2],
    },
    'karakter': {
        'AoV': [23.1, 0.0, 76.9],
        'Mobilelegends': [33.3, 33.3, 33.3],
        'Netdragons': [25.0, 0.0, 75.0],
    },
    'pakai': {
        'AoV': [71.4, 0.0, 28.6],
        'Mobilelegends': [100.0, 0.0, 0.0],
        'Netdragons': [36.4, 45.5, 18.2],
    },
}

navbar = dbc.NavbarSimple(
    brand="📱 Sentiment Analysis with Topic Modeling - LDA analysis output",
    color="primary",
    dark=True,
    )

body_layout = dbc.Container(
    [
    dbc.Row(
        [
        dbc.Col(
            [
            dcc.Markdown(
                f"""
                -----
                ##### Data:
                -----
                For this demonstration, {len(tSNE)} comments from the google play reviews were categorised into
                {len(tSNE.topic_id.unique())} topics using
                [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) analysis.

                Each topic is shown in different color on the citation map, as shown on the below.
                """
                )
            ],
            sm=12,
            md=6,
            ),
        dbc.Col([
            dcc.Markdown(
                """
                -----
                ##### LDA Hyperparameters
                -----
                Model hyperparameters can be thought of as settings for a machine learning algorithm that are tuned by the data scientist before training:
                * Number of Topics (K)
                * Dirichlet hyperparameter alpha: Document-Topic Density
                * Dirichlet hyperparameter beta: Word-Topic Density

                These hyperparameters will be evaluated using coherence values.

                """
                ),
            ],
            sm=12,
            md=6,
            ),

        ]
        ),

    dbc.Row(
        [
        dbc.Col(
            [
            dcc.Markdown(
                """
                -----
                ##### Coherence Score
                -----
                """
                ),
            dash_table.DataTable(
                coherence.to_dict('records'), [{"name": i, "id": i} for i in coherence.columns],
                style_table={'overflowX': 'auto'},
                )
            ],
            sm=12,
            md=6,
            ),
        dbc.Col(
            [
            dcc.Markdown(
                """
                -----
                ##### Topics:
                -----
                """
                ),
            html.Div(
                topics_html,
                style={
                "fontSize": 11,
                "overflow": "auto",
                },
                ),
            ],
            sm=12,
            md=6,
            ),
        ],
        ),

    dbc.Row(
        [
        dcc.Markdown(
            """
            -----
            ##### LDA Modeling
            -----
            """
            ),
        html.Iframe(src='assets/ldavis_moba9.html',
            className='w-100', height='750px'),
        ],
        ),

    dbc.Row(
        [
        dcc.Markdown(
            """
            -----
            ##### t-SNE test based on Perplexity
            -----
            """
            ),
        dcc.Tabs(
            [
            dcc.Tab(dcc.Graph(figure=tSNE1), label="t-SNE test, perplexity: 1"),
            dcc.Tab(dcc.Graph(figure=tSNE10), label="t-SNE test, perplexity: 10"),
            dcc.Tab(dcc.Graph(figure=tSNE25), label="t-SNE test, perplexity: 25"),
            dcc.Tab(dcc.Graph(figure=tSNE100), label="t-SNE test, perplexity: 100"),
            ]
            )
        ],
        ),

    dbc.Row(
        [
        dcc.Markdown(
            """
            -----
            ##### Filter data

            Use these filters to highlight reviews with:
            * application name, and
            * application sentiment

            -----
            """
            ),
        ]
        ),
    dbc.Row(
        [
        dbc.Col(
            [
            dbc.Card(
                dbc.CardBody(
                    [
                    html.H4("Sentiment apps by application name", className="card-title my-3"),
                    dcc.Dropdown(
                        id='categories-dropdown',
                        options=[{'label': apps, 'value': apps} for apps in data_app.keys()],
                        value='AoV'
                    ),
                    dcc.Graph(id='radar-app'),
                    ]
                    ),
                ),
            ],
            sm=12,
            md=6,
            ),
        dbc.Col(
            [
            dbc.Card(
                dbc.CardBody(
                    [
                    html.H4("Sentiment apps by topic", className="card-title my-3"),
                    dcc.Dropdown(
                        id='category-dropdown',
                        options=[{'label': categories, 'value': categories} for categories in data_topic.keys()],
                        value='aplikasi'
                    ),
                    dcc.Graph(id='radar-chart')
                    ]
                    ),
                ),
            ],
            sm=12,
            md=6,
            ),
        ]
        ),
    ],
    )

@app.callback(
    Output('radar-app', 'figure'),
    Input('categories-dropdown', 'value')
)

def update_radar_chart(selected_apps):
    fig = go.Figure()
    
    for trace, values in data_app[selected_apps].items():
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=trace
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True
    )
    return fig

@app.callback(
    Output('radar-chart', 'figure'),
    Input('category-dropdown', 'value')
)
def update_radar_chart(selected_category):
    fig = go.Figure()
    
    for trace, values in data_topic[selected_category].items():
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=sentiment,
            fill='toself',
            name=trace
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True
    )
    return fig

app.layout = html.Div([navbar, body_layout])

if __name__ == "__main__":
    app.run(debug=True)
