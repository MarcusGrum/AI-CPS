# syntax=docker/dockerfile:1
FROM python:3
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install matplotlib
Run pip3 install SciPy
Run pip3 install https://github.com/MarcusGrum/pybrain/archive/refs/heads/master.zip
Add ./../../code/annRequests_PyBrain /codeBase/