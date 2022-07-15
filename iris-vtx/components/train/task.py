import os
import click
import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


@click.command
@click.option('--data-path', type=str)
@click.option('--model-path', type=str)
def train(data_path: str, model_path: str):
    df = pd.read_csv(data_path, header=None)
    label = df.pop(df.shape[1] - 1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(label)

    # noinspection PyPep8Naming
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.3)
    ppl = make_pipeline(MinMaxScaler(), RandomForestClassifier())

    # noinspection PyUnresolvedReferences
    ppl.fit(X_train, y_train).predict(X_test)

    with open(model_path, 'wb') as f:
        pickle.dump(ppl, f)


if __name__ == '__main__':
    train()
