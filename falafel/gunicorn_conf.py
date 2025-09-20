#### Gunicorn configuration file
import multiprocessing
import os

import falafel.dl_model.main
import falafel.dl_model.constants as constants


#### Creates DL model befor workers are forked. Note that we do not load model here, because each worker should have its own model.
# def on_starting(server):

#     falafel.dl_model.main.main()
# 
    # if os.path.isfile(constants.MODEL_PATH):
    #     print('Model Found. Loading Model ...')
    # else:
    #     print('Model not Found. Creating Model ...')
    #     falafel.dl_model.main.main()

## For some reason, this code does not work
def on_starting(server):
    # if ( (constants.RST_MODEL_BOOL == 1) and (os.path.isfile(constants.MODEL_PATH)) ):       # This code doesnt enter loop, but doesnt load model correctly?
    #     os.remove(constants.MODEL_PATH)
    print('Starting Server ...')

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def on_exit(server):
    print('Exiting Server ...')
    



# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

#### Worker processes
workers = 1
# workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "eventlet"
# worker_class = 'sync'
# worker_connections = 1000
timeout = 600
# keepalive = 2

#### Server mechanics
# daemon = False
# pidfile = None
# umask = 0
# user = None
# group = None
# tmp_upload_dir = None

#### Logging
# errorlog = '-'
# loglevel = 'info'
# accesslog = '-'
# access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

#### Application module
wsgi_app = 'falafel.wsgi:create_app()'




#### TO RUN APP : gunicorn --config gunicorn.conf.py
   