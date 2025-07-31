import pickle
import networkx as nx
import pandas as pd

def load_graph(path="models/user_graph.gpickle"):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_single_user_features(G, user_row):
    user_id = user_row.get("user_id") or user_row.get("email")
    node = f"user_{user_id}"

    if node not in G:
        return {
            "num_connections": 0,
            "num_shared_identifiers": 0,
            "fraud_neighbors": 0,
            "fraud_ratio_neighbors": 0.0,
            "component_size": 1
        }

    neighbors = list(G.neighbors(node))
    fraud_neighbors = [n for n in neighbors if G.nodes[n].get("fraud") == 1 and n.startswith("user_")]
    component_size = len(nx.node_connected_component(G, node))

    return {
        "num_connections": len(neighbors),
        "num_shared_identifiers": len([n for n in neighbors if n.startswith("id_")]),
        "fraud_neighbors": len(fraud_neighbors),
        "fraud_ratio_neighbors": len(fraud_neighbors) / len(neighbors) if neighbors else 0,
        "component_size": component_size
    }

def get_connected_users(user_id, G):
    node = f"user_{user_id}"
    if node not in G:
        print(f"Node {node} not found in graph.")
        return []
    cluster = nx.node_connected_component(G, node)
    return [n.replace("user_", "") for n in cluster if n.startswith("user_") and n != node]
