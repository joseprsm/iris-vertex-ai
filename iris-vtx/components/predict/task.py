from typing import Optional

import os
import pickle

import numpy as np

from fastapi import FastAPI

from pydantic import BaseModel

from google.cloud import storage

app = FastAPI()

MODEL_URI = os.environ['MODEL_URI']
MODEL_PATH = MODEL_URI

print(MODEL_PATH)


with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


class Example(BaseModel):

    instances: list
    parameters: Optional[dict]


@app.post('/predict')
async def predict(example: Example):
    instances = np.array(example.instances).reshape(-1, 4)
    prediction = model.predict(instances)
    return {'predictions': prediction.astype(int).tolist()}


@app.get('/health')
async def health():
    return {'message': 'ok'}
