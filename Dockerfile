FROM python:3.10 AS base

COPY requirements.txt .

RUN pip install -r requirements.txt

FROM base AS train

RUN mkdir data outputs

RUN curl https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data --output data/iris.csv

COPY iris-vertex-ai/train.py train.py

ENTRYPOINT ["python", "train.py", "--data-path", "data/iris.csv"]
