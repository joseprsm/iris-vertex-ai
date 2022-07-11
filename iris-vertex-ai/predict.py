import os
import pickle
import numpy as np

from fastapi import FastAPI

from pydantic import BaseModel

from google.cloud import storage

app = FastAPI()

PROJECT_ID = os.environ.get("CLOUD_ML_PROJECT_ID", None)
BUCKET = os.environ.get('GCLOUD_BUCKET')

bucket = storage.Client(project=PROJECT_ID).bucket(BUCKET)
blob = bucket.blob('iris-test/model.pkl')
blob.download_to_filename('model.pkl')


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


class Example(BaseModel):

    instances: list
    parameters: dict


@app.post('/predict')
async def predict(example: Example):
    instances = np.array(example.instances).reshape(-1, 4)
    prediction = model.predict(instances)
    return {'predictions': prediction.astype(int).tolist()}


