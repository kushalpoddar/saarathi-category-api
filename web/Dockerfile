FROM python:3
RUN apt-get -y update
RUN apt-get install -y python3-pip build-essential ffmpeg libsm6 libxext6
COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 3201
ENTRYPOINT ["python3"]
CMD ["app.py"]