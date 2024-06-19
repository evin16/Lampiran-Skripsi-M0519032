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
app.title = "Sentiment analysis of applications based on LDA topic modeling"

df = pd.read_csv("dataset/dating-dataset.csv")
# Prep data / fill NAs
df["topic_id"] = df["topic_id"].astype(str)
topic_ids = [str(i) for i in range(len(df["topic_id"].unique()))]

# Topic Modeling LDA
with open("dataset/datings_topics.json", "r") as f:
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
coherence = pd.read_csv('dataset/lda_tuning_dating(1).csv')
coherence = coherence.sort_values(['Coherence'], ascending=[False])
coherence = coherence.head(10)

# t-SNE test based on perplexity
tSNE = pd.read_csv("dataset/dating-dataset.csv")
tSNE["topic_id"] = tSNE["topic_id"].astype(str)
tSNE1 = px.scatter(tSNE, x="x1", y="y1", color="topic_id", symbol="topic_id")
tSNE10 = px.scatter(tSNE, x="x10", y="y10", color="topic_id", symbol="topic_id")
tSNE25 = px.scatter(tSNE, x="x25", y="y25", color="topic_id", symbol="topic_id")
tSNE100 = px.scatter(tSNE, x="x100", y="y100", color="topic_id", symbol="topic_id")

# Sample data
apps = ['bumble', 'tinder']
categories = ['aplikasi', 'orang', 'saldo', 'masuk', 'akun', 'rumah', 'verifikasi']
sentiment = ['negative', 'neutral', 'positive']

data_app = {
    'bumble': {
        'negative': [46.3, 60.2, 37.0, 43.5, 53.5, 54.5, 65.4],
        'neutral': [20.3, 33.0, 40.7, 47.8, 44.2, 18.2, 30.8],
        'positive': [33,4, 6.8, 22.2, 8.7, 2.3, 27.3, 3.8],
    },
    'tinder': {
        'negative': [44.6, 58.6, 46.3, 73.6, 70.2, 46.4, 62.9],
        'neutral': [14.4, 23.5, 21.1, 13.6, 20.6, 23.2, 29.2],
        'positive': [41.0, 17.9, 32.7, 12.7, 9.2, 30.4, 7.9],
    },
}

data_topic = {
    'aplikasi': {
        'Bumble': [46.3, 20.3, 33.4],
        'Tinder': [44.6, 14.4, 41.0]
    },
    'orang': {
        'Bumble': [60.2, 33.0, 6.8],
        'Tinder': [58.6, 23.5, 17.9]
    },
    'saldo': {
        'Bumble': [37.0, 40.7, 22.2],
        'Tinder': [46.3, 21.1, 32.7]
    },
    'masuk': {
        'Bumble': [43.5, 47.8, 8.7],
        'Tinder': [73.6, 13.6, 12.7]
    },
    'akun': {
        'Bumble': [53.5, 44.2, 2.3],
        'Tinder': [70.2, 20.6, 9.2]
    },
    'rumah': {
        'Bumble': [54.5, 18.2, 27.3],
        'Tinder': [46.4, 23.2, 30.4]
    },
    'verifikasi': {
        'Bumble': [65.4, 30.8, 3.8],
        'Tinder': [62.9, 29.2, 7.9]
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
        html.Iframe(src='assets/ldavis_dating9.html',
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
                        value='bumble'
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
