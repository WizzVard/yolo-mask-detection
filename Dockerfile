FROM ultralytics/ultralytics:latest

ENV APP_HOME /home
WORKDIR ${APP_HOME}

# Copy the requirements file and install dependencies
COPY ./requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --no-cache-dir --upgrade -r ${APP_HOME}/requirements.txt

# Copy the model file
COPY ./MaskModel.pt ${APP_HOME}/MaskModel.pt

# Copy the app directory
# (the inference and api code is expected to be inside the app/ directory)
COPY ./app ${APP_HOME}/app

CMD ["fastapi", "run", "app/main.py", "--port", "8080"]