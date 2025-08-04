from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ensemble import predict_user_risk
from fake_profile import generate_fake_profile
from graph_feature import load_graph, get_connected_users
import pandas as pd
import random
import json

G=load_graph("models/user_graph.gpickle")

app = FastAPI()

df = pd.read_csv("Dataset/Base_with_graph_features.csv")
df = df.drop(columns=["fraud_bool"], errors="ignore")
df = df.dropna(subset=["device_os", "source", "email", "phone_number", "device_id", "ip_address"])

class InputData(BaseModel):
    data: dict

with open("top_risky_preview.json") as f:
    cached_preview = json.load(f)

@app.get("/users")
def get_summary():
    return {
        "total_users": len(df),
        "top_risky_users": cached_preview
    }

@app.get("/user_profile/{user_id}")
def get_user_profile(user_id: str):
    return generate_fake_profile(user_id)

@app.post("/predict")
def predict(data: InputData):
    required_fields = ["email", "phone_number", "device_id", "ip_address", "device_os", "source"]
    for col in required_fields:
        if col not in data.data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {col}")
    try:
        user_df = pd.DataFrame([data.data])
        result = predict_user_risk(user_df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fraud_cluster/{user_id}")
def get_fraud_cluster(user_id: str):
    connected = get_connected_users(user_id, G)
    return {"connected_users": connected}
