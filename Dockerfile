FROM python:3.10 AS base

COPY requirements.txt .

RUN pip install -r requirements.txt

FROM base AS train

ARG BUCKET
ENV GCLOUD_BUCKET=$BUCKET

RUN mkdir data outputs

COPY iris-vtx/components/train/task.py train.py

ENTRYPOINT ["python", "train.py"]

FROM base AS predict

ARG BUCKET
ENV GCLOUD_BUCKET=$BUCKET

COPY iris-vtx/components/predict/task.py predict.py

ENTRYPOINT ["uvicorn", "predict:app", "--reload", "--host=0.0.0.0"]

FROM python:3.10 AS deploy

RUN pip install click google-cloud-aiplatform

COPY iris-vtx/components/deploy/task.py deploy.py
