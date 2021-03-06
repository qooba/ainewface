#!/bin/bash

cp -r ../src/app ./app
cp -r ../models ./models
docker build -t qooba/ainewface .
docker build -t qooba/ainewface:dev -f Dockerfile.dev . 
rm -rf ./app ./models

