import os
import click
import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# noinspection PyPackageRequirements
from google.cloud import storage

BUCKET = os.environ['GCLOUD_BUCKET']
OUTPUT_DIR = 'outputs'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.pkl')

bucket = storage.Client().bucket(BUCKET)
blob = bucket.blob('iris-test/model.pkl')


@click.command
@click.option('--data-path', default='data/iris.csv')
def train(data_path: str):
    df = pd.read_csv(data_path, header=None)
    label = df.pop(df.shape[1] - 1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(label)

    # noinspection PyPep8Naming
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.3)
    ppl = make_pipeline(MinMaxScaler(), RandomForestClassifier())

    # noinspection PyUnresolvedReferences
    ppl.fit(X_train, y_train).predict(X_test)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(ppl, f)

    blob.upload_from_filename(MODEL_PATH)


if __name__ == '__main__':
    train()
