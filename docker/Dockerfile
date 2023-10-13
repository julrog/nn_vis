FROM python:3.9

RUN apt-get -y update
RUN apt-get -y install mesa-utils

WORKDIR /workspace/

RUN git clone https://github.com/julrog/nn_vis.git

WORKDIR /workspace/nn_vis

RUN pip install -r requirements.txt

COPY docker/prepare_sample_data.py .
RUN python prepare_sample_data.py

CMD python start_tool.py --demo
