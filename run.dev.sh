#!/bin/bash
docker run --gpus all -d --name fake_generator --rm -v $(pwd)/src/app:/app -v $(pwd)/notebooks:/opt/notebooks -p 8888:8888 -p 8000:8000 qooba/tf
