FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

WORKDIR /workspace/nn_vis

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN sudo apt-get -y update
RUN sudo apt-get -y install mesa-utils

RUN sudo pip install pre-commit
