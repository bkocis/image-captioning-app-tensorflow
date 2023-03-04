FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

COPY ./application /opt/application
COPY ./models /opt/models
COPY ./utils /opt/utils
COPY ./requirements.txt /opt/requirements.txt

RUN mkdir -p /opt/resources/
RUN apt-get update
RUN apt-get install python3-dev -y

RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip install -r /opt/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

ENV PYTHONPATH /opt
WORKDIR /opt
EXPOSE 8081

CMD ["gunicorn", "application.main:app", "-c", "application/config/gunicorn_config.py", "-k", "uvicorn.workers.UvicornWorker"]