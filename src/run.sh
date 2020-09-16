#!/usr/bin/env bash
docker build . -t sps
for f in ./train_data/* 
do
 filename=$(basename -- "$f")
 echo $filename: $(docker run sps $filename)
done
