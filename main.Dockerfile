FROM python:3.10.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 5000