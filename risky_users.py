import pandas as pd
from ensemble import predict_user_risk

df = pd.read_csv("Dataset/Base_with_graph_features.csv")
df = df.drop(columns=["fraud_bool"], errors="ignore")
df = df.dropna(subset=["device_os", "source", "email", "phone_number", "device_id", "ip_address"])

sample_df = df.sample(100, random_state=42).copy()
sample_df["score"] = sample_df.apply(lambda x: predict_user_risk(pd.DataFrame([x]))["ensemble_score"], axis=1)

top5 = sample_df.sort_values("score", ascending=False).head(5)
preview = top5[["email", "score"]].reset_index(drop=True)

preview.to_json("top_risky_preview.json", orient="records", indent=2)
