FROM python:3.10 AS base

COPY requirements.txt .

RUN pip install -r requirements.txt

FROM base AS train

RUN mkdir data outputs

COPY iris-vtx/components/train/task.py train.py

ENTRYPOINT ["python", "train.py"]

FROM base AS predict

COPY iris-vtx/components/predict/task.py predict.py

ENTRYPOINT ["uvicorn", "predict:app", "--reload", "--host=0.0.0.0", "--port", "8080"]

FROM python:3.10 AS deploy

RUN pip install click google-cloud-aiplatform

ARG SERVICE_ACCOUNT
ENV SERVICE_ACCOUNT=$SERVICE_ACCOUNT

COPY iris-vtx/components/deploy/task.py deploy.py
