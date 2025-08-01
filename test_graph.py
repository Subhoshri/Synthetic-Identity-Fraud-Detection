from graph_feature import load_graph, extract_single_user_features, get_connected_users

G = load_graph("models/user_graph.gpickle")

test_user = {"user_id": "user1467@example.com"}

features = extract_single_user_features(G, test_user)
print("Graph features for user:")
print(features)

connected = get_connected_users(test_user["user_id"], G)
print(f"Connected fraud cluster for {test_user['user_id']}:")
print(connected)
