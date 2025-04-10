import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import networkx as nx

# Load the pre-built graph
G = nx.read_gpickle("supply_chain_graph.gpickle")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Supply Chain Network Visualization"),
    html.Div([
        html.Label("Search for a Company:"),
        dcc.Input(id="search-input", type="text", placeholder="Enter company name...", style={'width': '50%'}),
    ]),
    dcc.Graph(id="graph-visualization")
])

def generate_figure(subgraph):
    pos = nx.spring_layout(subgraph, k=0.3, iterations=20)
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} (Degree: {subgraph.degree(node)})")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=10,
            color='blue',
            line_width=2
        )
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Supply Chain Network Graph",
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                    ))
    return fig

@app.callback(
    Output("graph-visualization", "figure"),
    Input("search-input", "value")
)
def update_graph(search_value):
    if search_value:
        filtered_nodes = [n for n in G.nodes if search_value.lower() in n.lower()]
        if not filtered_nodes:
            return go.Figure(data=[], layout=go.Layout(title="No matching companies found."))
        subgraph = G.subgraph(filtered_nodes)
    else:
        subgraph = G
    return generate_figure(subgraph)

if __name__ == "__main__":
    app.run_server(debug=True)
