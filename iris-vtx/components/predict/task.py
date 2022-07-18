from typing import Optional

import os
import pickle

import numpy as np

from fastapi import FastAPI

from pydantic import BaseModel

from google.cloud import storage

app = FastAPI()

MODEL_URI = os.environ.get('MODEL_URI')
BUCKET_NAME = MODEL_URI.split('//')[1].split('/')[0]
BLOB_LOC = os.path.join('/'.join(MODEL_URI.split('//')[1].split('/')[1:]), 'model.pkl')

bucket = storage.Client().bucket(BUCKET_NAME)
blob = bucket.blob('/'.join(BLOB_LOC))
blob.download_to_filename('model.pkl')


with open('model.pkl', 'rb') as f:
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
