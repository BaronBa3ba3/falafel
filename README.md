# falafel

### Neuron Network for plant recognition

### Architecture 

`Client -> NGINX -> WSGI -> Flask -> DL_Model`


### Testing Server (Client-Side)


Website :   `localhost:5000`

CLI :       `curl -X POST -F "image=@XXX.jpg" http://localhost:5000/predict`
            `curl -X POST -F "file=@XXX.jpg" http://localhost:5000/predict`


NGINX listens on port 5000
WSGI  listens on port 8000



### Testing Server (Server-Side)

#### Updating Docker container

`docker build -t ba3ba3/falafel:vX.XX .`
`docker push ba3ba3/falafel:vX.XX`
docker push ba3ba3/falafel:v1.00

#### Lauching Gunicorn app

`gunicorn --config falafel/gunicorn_conf.py`



### Steps to install application onto Ubuntu Server (Server-Side)

#### Docker Compose

1. Download compose.yaml

2. `docker compose up`


#### Direct Method

1. Create working directory and enter it (suggestion : `/etc/falafel`)

- `mkdir /etc/falafel`
- `cd /etc/falafel`

2. Copy the Application to the Server:

Transfer your application package to the server using scp, rsync, or any other method.


3. Run "`./scripts/install_falafel.sh`  or `bash scripts/install_falafel.sh`

If it doesn't work, make sure that the script is executable : `chmod a+x /scripts/install_falafel.sh`


### Application Structure


falafel/
├── config/
│   ├── falafel.conf
│   └── nginx.conf
|
├── falafel/
│   ├── __init__.py
│   ├── app.py
│   ├── gunicorn_conf.py
│   ├── wsgi.py
│   |
│   ├── templates/
│   |   ├── index.html
│   |   └── model.html
|   |   
│   └── dl_model/
│       ├── __init__.py
│       ├── main.py
│       ├── constants.py
│       ├── getDatabse.py
│       ├── createModel.py
│       ├── trainModel.py
|       └── predict.py
|
├── Dockerfile
├── requirements.txt
└── docker-compose.yml



