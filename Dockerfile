FROM python:3.10 AS base

COPY requirements.txt .

RUN pip install -r requirements.txt

FROM base AS train

ARG BUCKET
ENV GCLOUD_BUCKET=$BUCKET

RUN mkdir data outputs

RUN curl https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data --output data/iris.csv

COPY components/train/task.py train.py

ENTRYPOINT ["python", "train.py", "--data-path", "data/iris.csv"]

FROM base AS predict

ARG BUCKET
ENV GCLOUD_BUCKET=$BUCKET

COPY components/predict/task.py predict.py

ENTRYPOINT ["uvicorn", "predict:app", "--reload", "--host=0.0.0.0"]