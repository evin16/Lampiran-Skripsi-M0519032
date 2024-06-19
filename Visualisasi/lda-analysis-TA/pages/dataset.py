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

df = pd.read_csv("output/dating-dashboard.csv")
# Prep data / fill NAs
df["topic_id"] = df["topic_id"].astype(str)
topic_ids = [str(i) for i in range(len(df["topic_id"].unique()))]

# Topic Modeling LDA
with open("output/lda_topics.json", "r") as f:
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

    @server.route('/LDA_model')
    def LDA_model():
        return render_template('output/lda_vis.html')


# Topic view
topics_html = list()
for topic_html in [
    html.Span([str(i) + ": " + topics_txt[i]], style={"color": col_swatch[i]})
    for i in range(len(topics_txt))
    ]:
    topics_html.append(topic_html)
    topics_html.append(html.Br())

# Coherence Model
coherence = pd.read_csv('output/lda_tuning_results.csv')
coherence = coherence.sort_values(['Coherence'], ascending=[False])
coherence = coherence.head(10)

# t-SNE test based on perplexity
tSNE = pd.read_csv("output/lda_df.csv")
tSNE["topic_id"] = tSNE["topic_id"].astype(str)
tSNE1 = px.scatter(tSNE, x="x1", y="y1", color="topic_id", symbol="topic_id")
tSNE10 = px.scatter(tSNE, x="x10", y="y10", color="topic_id", symbol="topic_id")
tSNE25 = px.scatter(tSNE, x="x25", y="y25", color="topic_id", symbol="topic_id")
tSNE75 = px.scatter(tSNE, x="x75", y="y75", color="topic_id", symbol="topic_id")

# Count
count = pd.read_csv('output/count.csv')
count['topic'] = count['topic'].apply(str)

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
        html.Iframe(src='assets/lda_vis.html',
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
            dcc.Tab(dcc.Graph(figure=tSNE75), label="t-SNE test, perplexity: 75"),
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
                        id="dropdown-app",
                        options=["bumble", "tinder"],
                        value="bumble",
                        clearable=False,
                        ),
                    dcc.Graph(id="graph-app"),
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
                        id="dropdown-topic",
                        options=["0", "1", '2', '3', '4', '5', '6', '7', '8'],
                        value="0",
                        clearable=False,
                        ),
                    dcc.Graph(id="graph-topic"),
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
    Output("graph-app", "figure"),
    Input("dropdown-app", "value"))

def update_bar_app(app):
    mask = count["aplikasi"] == app
    fig = px.bar(count[mask], x="sentiment", y="value",
       color="topic", barmode="group")
    return fig

@app.callback(
    Output("graph-topic", "figure"),
    Input("dropdown-topic", "value"))

def update_bar_topic(topic):
    mask = count["topic"] == topic
    fig = px.bar(count[mask], x="sentiment", y="value",
        color="aplikasi", barmode="group")
    return fig

app.layout = html.Div([navbar, body_layout])

if __name__ == "__main__":
    app.run(debug=True)
