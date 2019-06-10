#!/bin/bash

sudo apt-get install python-dev python-pip git docker.io
sudo pip install virtualenv
sudo apt-get install docker-ce docker-ce-li containerd.io
python3 -m venv WebApp/venv
cd WebApp
source venv/bin/activate

sudo docker build -t cnnapp:latest .
sudo docker stop containercnn
sudo docker rm containercnn
sudo service docker restart
docker run -d --name containercnn --restart=always -p 5000:5000 cnnapp
