# Medisearch-api
# Getting Started with this project

## How to start

In the project directory, you can run:

### `pip install -r requirements.txt`

Runs the app.
### `uvicorn "app.main:app --reload"`
Open [http://localhost:8000](http://localhost:8000) to view it in your browser.

The page will reload when you make changes.




## How to create image using Dockerfile

Build the image locally.\
`docker build -t "medisearch:latest" .`

or you can pull the image from docker registry using this command:\
`docker pull hakimamarullah/medisearch:latest`


### How to run using docker container in detached mode:\
`docker run --name $container_name -p $port:80 --restart $restart-policy -d hakimamarullah/medisearch:latest`
