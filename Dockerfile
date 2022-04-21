# 1、 From the official  Python  The basic image begins 
From python:3.6.8 as builder

# 2、 Set the current working directory to  /code

#  This is placement  requirements.txt  Where files and application directories are located 
RUN mkdir model
WORKDIR /model

COPY . /model
COPY ./requirements.txt /model/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /model/requirements.txt

CMD ["python3", "_fast_.py"]
EXPOSE 2222
