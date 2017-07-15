#!/bin/bash

for id in `docker ps -a| tail -n +2 | awk -F " " '{print $1}'`; do docker rm  $id; done
docker build -t tf .
#docker run -it -p 8888:8888 tf
cwd=`PWD`
docker run -it -p 8888:8888 -p 6006:6006 -v  $cwd/notebooks:/notebooks -v $cwd/data:/data tf