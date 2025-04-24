import json
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Load spaCy English model; consider a domain-specific model if available
nlp = spacy.load("en_core_web_sm")

G = nx.Graph()

# Load the scraped data
with open("/Users/c27ds/Dev/AdvCs/sprint1/supplychain_project/supply_chain_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each article and extract organizations (ORG)
for entry in data:
    if "content" in entry and entry["content"]:
        doc = nlp(entry["content"])
        companies = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
        companies = list(set([company.lower() for company in companies if company]))
        # Add each company as a node and add edges for co-occurrences
        for comp in companies:
            G.add_node(comp)
        for i in range(len(companies)):
            for j in range(i+1, len(companies)):
                if G.has_edge(companies[i], companies[j]):
                    G[companies[i]][companies[j]]['weight'] += 1
                else:
                    G.add_edge(companies[i], companies[j], weight=1)

# (Optional) Save the graph to a file
with open("supply_chain_graph.gpickle", "wb") as f:
    pickle.dump(G, f)

# For debugging: Visualize a simple layout of the graph
if G.number_of_nodes() > 0:
    pos = nx.spring_layout(G, k=0.3, iterations=20)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, width=weights, node_size=500, font_size=10)
    plt.title("Supply Chain Network (Extracted Entities)")
    plt.show()
else:
    print("Graph is empty. Check your entity extraction!")
