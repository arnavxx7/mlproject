FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt /app
RUN apt update -y
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["python3", "app.py"]
