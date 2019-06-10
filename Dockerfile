FROM tiangolo/uwsgi-nginx-flask:python3.6
WORKDIR /app
RUN apt-get update
COPY requirements.txt /app
RUN pip install -U pip
RUN pip install -r ./requirements.txt
COPY server.py /app
COPY . /app

EXPOSE 5000
ENV FLASK_APP server.py
CMD ["flask", "run", "--host", "0.0.0.0"]