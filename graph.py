import pandas as pd
import networkx as nx
import pickle

df = pd.read_csv("Dataset/Base_with_identifiers.csv")
if "user_id" not in df.columns:
    df["user_id"]=df.index
df.to_csv("Dataset/Base_with_identifiers.csv", index=False)
G = nx.Graph()

for idx, row in df.iterrows():
    G.add_node(f"user_{row["user_id"]}", fraud=row["fraud_bool"])

identifier_cols = ["email", "phone_number", "device_id", "ip_address"]

for col in identifier_cols:
    shared = df.groupby(col)["user_id"].apply(list)
    for user_list in shared:
        if len(user_list) > 1:
            identifier_node = f"id_{col}_{user_list[0]}"
            G.add_node(identifier_node)
            for uid in user_list:
                G.add_edge(f"user_{uid}", identifier_node, link=col)

with open("models/user_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
