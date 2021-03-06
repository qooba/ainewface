#!/bin/bash
docker run --gpus all -d --name fake_generator --rm -p 8000:8000 qooba/ainewface
