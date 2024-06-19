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

df = pd.read_csv("dataset/social-dataset.csv")
# Prep data / fill NAs
df["topic_id"] = df["topic_id"].astype(str)
topic_ids = [str(i) for i in range(len(df["topic_id"].unique()))]

# Topic Modeling LDA
with open("dataset/topics_social.json", "r") as f:
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
coherence = pd.read_csv('dataset/lda_tuning_social(1).csv')
coherence = coherence.sort_values(['Coherence'], ascending=[False])
coherence = coherence.head(10)

# t-SNE test based on perplexity
tSNE = pd.read_csv("dataset/social-dataset.csv")
tSNE["topic_id"] = tSNE["topic_id"].astype(str)
tSNE1 = px.scatter(tSNE, x="x1", y="y1", color="topic_id", symbol="topic_id")
tSNE10 = px.scatter(tSNE, x="x10", y="y10", color="topic_id", symbol="topic_id")
tSNE25 = px.scatter(tSNE, x="x25", y="y25", color="topic_id", symbol="topic_id")
tSNE100 = px.scatter(tSNE, x="x100", y="y100", color="topic_id", symbol="topic_id")

# Sample data
apps = ['facebook', 'instagram', 'tiktok']
categories = ['aplikasi', 'akun', 'tik', 'fitur', 'manusia', 'anak']
sentiment = ['negative', 'neutral', 'positive']

data_app = {
    'facebook': {
        'negative': [45.2, 47.1, 51.2, 56.8, 61.6, 39.7],
        'neutral': [34.2, 46.0, 26.2, 30.9, 17.0, 33.3],
        'positive': [20.6, 6.9, 22.6, 12.3, 21.4, 27.0],
    },
    'instagram': {
        'negative': [58.1, 51.4, 67.7, 60.0, 64.2, 60.2],
        'neutral': [30.7, 40.5, 22.6, 33.3, 20.1, 27.3],
        'positive': [11.1, 8.1, 9.7, 6.7, 15.7, 12.5],
    },
    'tiktok': {
        'negative': [22.4, 32.9, 23.5, 41.6, 38.5, 18.8],
        'neutral': [24.4, 29.9, 20.3, 45.0, 14.6, 21.8],
        'positive': [53.2, 37.2, 56.2, 13.4, 46.9, 59.4],
    },
}

data_topic = {
    'aplikasi': {
        'Facebook': [45.2, 34.2, 20.6],
        'Instagram': [58.1, 30.7, 11.1],
        'Tiktok': [22.4, 24.4, 53.2],
    },
    'akun': {
        'Facebook': [47.1, 46.0, 6.9],
        'Instagram': [51.4, 40.5, 8.1],
        'Tiktok': [32.9, 29.9, 37.2],
    },
    'tik': {
        'Facebook': [51.2, 26.2, 22.6],
        'Instagram': [67.7, 22.6, 9.7],
        'Tiktok': [23.5, 20.3, 56.2],
    },
    'fitur': {
        'Facebook': [56.8, 30.9, 12.3],
        'Instagram': [60.0, 33.3, 6.7],
        'Tiktok': [41.6, 45.0, 13.4],
    },
    'manusia': {
        'Facebook': [61.6, 17.0, 21.4],
        'Instagram': [64.2, 20.1, 15.7],
        'Tiktok': [38.5, 14.6, 46.9],
    },
    'anak': {
        'Facebook': [39.7, 33.3, 27.0],
        'Instagram': [60.2, 27.3, 12.5],
        'Tiktok': [18.8, 21.8, 59.4],
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
        html.Iframe(src='assets/ldavis_social8.html',
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
                        value='facebook'
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
