from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = FastAPI(title="Swiggy ETA Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('mimo_model.keras')
scaler = joblib.load('scaler.pkl')
EXPECTED_COLUMNS = joblib.load('expected_columns.pkl')

class OrderRequest(BaseModel):
    driver_age: int
    driver_rating: float
    distance_km: float
    weather: str
    traffic: str
    vehicle_type: str
    festival: str
    city_type: str

@app.post("/predict")
async def predict_eta(order: OrderRequest):
    try:
        input_df = pd.DataFrame(0.0, index=[0], columns=EXPECTED_COLUMNS)
        if 'Delivery_person_Age' in input_df.columns:
            input_df['Delivery_person_Age'] = float(order.driver_age)
        if 'Delivery_person_Ratings' in input_df.columns:
            input_df['Delivery_person_Ratings'] = float(order.driver_rating)
        if 'distance_km' in input_df.columns:
            input_df['distance_km'] = float(order.distance_km)

        categories = [order.weather, order.traffic, order.vehicle_type, order.city_type]
            
        if order.festival == "Yes":
            categories.append("Festival_Yes")

        for col in input_df.columns:
            for cat in categories:
                if cat.lower() in col.lower():
                    input_df[col] = 1.0

        scaled_features = scaler.transform(input_df.values)
        predictions = model.predict(scaled_features)
        
        prep_time = max(0, float(predictions[0][0][0]))
        travel_time = max(0, float(predictions[1][0][0]))

        print(f"Age: {order.driver_age} | Raw Prep: {prep_time} | Raw Travel: {travel_time}")
        
        return {
            "status": "success",
            "eta_breakdown": {
                "prep_time_minutes": round(prep_time, 1),
                "travel_time_minutes": round(travel_time, 1),
                "total_eta_minutes": round(prep_time + travel_time, 1)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))