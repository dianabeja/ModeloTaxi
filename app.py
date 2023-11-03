from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='Insurance Premium Prediction')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained regression model
model = load(pathlib.Path('model/train-regression-v1.joblib'))


class InputData(BaseModel):
    distance_traveled: int = 44
    num_of_passengers: int = 1
    miscellaneous_fees: float = 28.0
    trip_duration: float = 0.0

class OutputData(BaseModel):
    fare: float

@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    
    # Prepare the input for prediction
    input_features = [
        data.trip_duration,
        data.distance_traveled,
        data.num_of_passengers,
        data.miscellaneous_fees,
    ]

    # Make the prediction using the model
    result = model.predict([input_features])

    return OutputData(fare=result[0])