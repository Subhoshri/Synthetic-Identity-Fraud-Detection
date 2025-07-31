import joblib
import numpy as np
import pandas as pd
from rules import apply_rules
from graph_feature import load_graph, extract_single_user_features

scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")
rf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")

with open("feature_order.txt") as f:
    feature_order = f.read().splitlines()

G = load_graph("models/user_graph.gpickle")

WEIGHTS = {
    "rf": 0.3,
    "iso": 0.3,
    "rules": 0.3,
    "graph": 0.1
}
THRESHOLD = 0.30

cat = ["device_os", "source"]
num = [col for col in feature_order if col not in encoder.get_feature_names_out(cat)]

def predict_user_risk(user_row_df):
    encoded_cat = encoder.transform(user_row_df[cat])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat))

    combined = pd.concat([user_row_df[num].reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)
    combined = combined[feature_order]

    X = scaler.transform(combined)
    X = pd.DataFrame(X, columns=feature_order)
    
    X_np=X.to_numpy()
    rf_score = rf_model.predict_proba(X_np)[0][1]
    iso_raw = iso_model.decision_function(X_np)[0]
    iso_score = 1 - (iso_raw - iso_model.offset_)

    rule_score, reasons = apply_rules(user_row_df.iloc[0])

    graph_feats = extract_single_user_features(G, user_row_df.iloc[0])
    graph_score = min(1.0, graph_feats.get("fraud_ratio_neighbors", 0.0) + graph_feats.get("fraud_neighbors", 0)/5)

    final_score = (
        WEIGHTS["rf"] * rf_score +
        WEIGHTS["iso"] * iso_score +
        WEIGHTS["rules"] * rule_score +
        WEIGHTS["graph"] * graph_score
    )

    is_fraud = int(final_score >= THRESHOLD)

    return {
        "ensemble_score": round(final_score, 3),
        "is_fraud": is_fraud,
        "rf_score": round(rf_score, 3),
        "iso_score": round(iso_score, 3),
        "rule_score": round(rule_score, 3),
        "graph_score": round(graph_score, 3),
        "reasons": reasons
    }
