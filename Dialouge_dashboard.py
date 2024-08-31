import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.data.path.append('C:\\nltk_data')
nltk.download('punkt_tab', download_dir='C:\\nltk_data')
nltk.download('stopwords', download_dir='C:\\nltk_data')
              



movie_lines = pd.DataFrame({
    'lineID': range(1, 101),
    'characterID': np.random.randint(1, 11, 100),
    'movieID': np.random.randint(1, 6, 100),
    'character': [f'Character_{i}' for i in np.random.randint(1, 11, 100)],
    'text': [f'This is sample dialogue number {i}' for i in range(1, 101)]
})

movie_conversations = pd.DataFrame({
    'characterID1': np.random.randint(1, 11, 50),
    'characterID2': np.random.randint(1, 11, 50),
    'movieID': np.random.randint(1, 6, 50),
    'utteranceIDs': [str(list(np.random.choice(range(1, 101), size=np.random.randint(2, 6)))) for _ in range(50)]
})

# Preprocess text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

movie_lines['processed_text'] = movie_lines['text'].apply(preprocess_text)

# Create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dialogue-Based Personal Assistant Dashboard", className="text-center mb-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='conversation-length-histogram')
        ], width=6),
        dbc.Col([
            dcc.Graph(id='word-frequency-barchart')
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='character-interaction-network')
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.H4("Dialogue Simulation", className="text-center"),
            dcc.Input(id='user-input', type='text', placeholder='Enter your message...', style={'width': '100%'}),
            html.Button('Send', id='send-button', n_clicks=0, className="mt-2 btn btn-primary"),
            html.Div(id='assistant-response', className="mt-3 p-3 border rounded")
        ], width=12)
    ])
], fluid=True)

# Callback for conversation length histogram
@app.callback(
    Output('conversation-length-histogram', 'figure'),
    Input('conversation-length-histogram', 'id')
)
def update_conversation_length_histogram(_):
    conversation_lengths = movie_conversations['utteranceIDs'].apply(lambda x: len(eval(x)))
    fig = px.histogram(x=conversation_lengths, nbins=30, title="Distribution of Conversation Lengths")
    fig.update_layout(xaxis_title="Number of Utterances", yaxis_title="Frequency")
    return fig


@app.callback(
    Output('word-frequency-barchart', 'figure'),
    Input('word-frequency-barchart', 'id')
)
def update_word_frequency_barchart(_):
    all_words = [word for words in movie_lines['processed_text'] for word in words]
    word_freq = Counter(all_words).most_common(20)
    df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
    fig = px.bar(df, x='word', y='frequency', title="Top 20 Most Frequent Words")
    fig.update_layout(xaxis_title="Word", yaxis_title="Frequency")
    return fig

# Callback for word frequency bar chart
import plotly.graph_objs as go
import networkx as nx
from plotly.subplots import make_subplots

@app.callback(
    Output('character-interaction-network', 'figure'),
    Input('character-interaction-network', 'id')
)
def update_character_interaction_network(_):
    character_interactions = movie_conversations[['characterID1', 'characterID2']].value_counts().reset_index()
    character_interactions.columns = ['source', 'target', 'weight']

    # Create a graph
    G = nx.Graph()
    for _, row in character_interactions.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    # Use Fruchterman-Reingold layout for node positioning
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Calculate node sizes based on degree
    node_sizes = [v * 20 + 20 for v in dict(G.degree()).values()]

    # Calculate edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)

    # Custom color palette
    node_color = '#3a506b'
    edge_color = '#1c2541'
    background_color = '#ffffff'

    # Create edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color=edge_color),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Create node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (f'Character {node}',)

    # Color node points by the number of connections
    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = [f'Character {node}<br># of connections: {adj}'
                       for node, adj in zip(G.nodes(), node_adjacencies)]

    # Create the figure
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Character Interaction Network'])
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    fig.update_layout(
        title='Character Interaction Network',
        title_x=0.5,
        titlefont_size=18,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[dict(
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=background_color,
        paper_bgcolor=background_color
    )

    return fig

# Callback for dialogue simulation
@app.callback(
    Output('assistant-response', 'children'),
    Input('send-button', 'n_clicks'),
    State('user-input', 'value')
)
def update_assistant_response(n_clicks, user_input):
    if n_clicks > 0 and user_input:
        # Simple response generation (to be replaced with actual RL model)
        response = "I understand you said: '" + user_input + "'. As an AI assistant, I'm here to help. How can I assist you further?"
        return response
    return "Assistant is ready to chat. Type a message and click 'Send'."

if __name__ == '__main__':
    app.run_server(debug=True)