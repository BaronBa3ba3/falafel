# falafel

Server hosting Neuron Network for plant recognition



## Architecture 

`Client -> NGINX -> WSGI -> Flask`




## Steps to install application

### Install with Docker Compose (Recommended)
This Method is recommended because it allows to set a reverse proxy (NGINX) in front of the GUNICORN server


1. Create directory
- `mkdir falafel`
- `cd falafel`

2. Clone Repo from Github
`git clone https://github.com/BaronBa3ba3/falafel.git`

3. Edit configs files
`nano config/falafel.conf`
`nano config/nginx.conf`

4. Build Docker Containers
`docker compose up`


### Install with Docker
This method is less secure and efficient, as GUNICORN's purpose is not to handle http requests




### Install the application directly onto system (not recommended):

1. Create working directory and enter it (suggestion : `/etc/falafel`)

- `mkdir /etc/falafel`
- `cd /etc/falafel`

2. Copy the Application to the Server:

Transfer your application package to the server using scp, rsync, or any other method.


3. Run "`./scripts/install_falafel.sh`  or `bash scripts/install_falafel.sh`

If it doesn't work, make sure that the script is executable : `chmod a+x /scripts/install_falafel.sh`




## Ideas to integrate

When receives several images of same plant, uses an algorithm (maybe sums all prediction of same plant) that considers the predictions of all images of the same plant and determines most likely result.8

