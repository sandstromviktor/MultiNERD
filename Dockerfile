FROM python:3.8-slim

ARG HOME=/home/code
COPY . $HOME

RUN apt update -y \ 
    && apt upgrade -y

RUN pip install --no-cache --upgrade pip \
    && pip install -r $HOME/requirements.txt

WORKDIR $HOME