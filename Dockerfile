FROM python:3.11.5

WORKDIR /

RUN pip install --upgrade pip

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY .data/* ./data/
COPY ./app/predict.py ./app/
COPY ./app/test.py /app/

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=9696", "app.predict:app"]