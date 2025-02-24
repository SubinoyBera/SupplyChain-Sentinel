FROM python:3.11.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y
COPY . /app

WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
