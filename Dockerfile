FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app
COPY requirements.txt /app
COPY artifacts /app
RUN apt update -y
RUN pip install scikit-learn==1.6.1 dill
RUN pip install -r requirements.txt


EXPOSE 8080

CMD ["python3", "app.py"]
