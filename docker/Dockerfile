FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter 
RUN pip3 install qdi fastapi uvicorn jinja2 aiofiles python-multipart
ENV PYTHONPATH="/app:${PYTHONPATH}"
WORKDIR /app
COPY ./app /app
COPY ./models /models
ENTRYPOINT [ "/bin/bash", "./start-reload.sh" ]
