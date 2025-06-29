import json
import networkx as nx
import matplotlib.pyplot as plt
import random

# --- Load SVO triplets ---
with open("data/svo_triplets.json") as f:
    triplets = json.load(f)

# --- Build Knowledge Graph with verbs as edge labels ---
G = nx.DiGraph()
for t in triplets:
    subj = t["subject"]
    obj = t["object"]
    verb = t["verb"]
    G.add_edge(subj, obj, label=verb)

print(f"[✓] Knowledge Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --- Draw Sampled Subgraph with More Edges & Verbs ---
def draw_sampled_kg(G, limit=300):
    sampled_edges = random.sample(list(G.edges(data=True)), min(limit, G.number_of_edges()))
    subG = nx.DiGraph()
    subG.add_edges_from(sampled_edges)

    plt.figure(figsize=(28, 28))  # Bigger figure for more content
    pos = nx.spring_layout(subG, k=0.5)

    nx.draw_networkx_nodes(subG, pos, node_color="lightblue", node_size=900)
    nx.draw_networkx_edges(subG, pos, edge_color="gray", arrows=True, arrowsize=20, width=1.5)
    nx.draw_networkx_labels(subG, pos, font_size=9, font_color="black")

    edge_labels = nx.get_edge_attributes(subG, 'label')
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=8, font_color="darkred", rotate=False)

    plt.title("Sampled Knowledge Graph with Verb Edge Labels", fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Call the drawing function ---
draw_sampled_kg(G, limit=300)
