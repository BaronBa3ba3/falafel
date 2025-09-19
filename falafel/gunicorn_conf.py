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

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

#### Worker processes
workers = 1
# workers = multiprocessing.cpu_count() * 2 + 1
# worker_class = 'sync'
# worker_connections = 1000
# timeout = 30
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
   