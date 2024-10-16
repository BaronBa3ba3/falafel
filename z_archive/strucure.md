# Application Structure :


### Docker Structure

`
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
`


This Structure is used for Docker Build.



### Manual Installation Structure

`
falafel/
│
├── falafel/
│   ├── __init__.py
│   ├── app.py                      # Main application code
│   └── wsgi.py                     # WSGI entry point
│
├── scripts/
│   ├── install_falafel.sh          # Installation script
│   └── start_gunicorn.sh           # Script to start Gunicorn
│
├── setup.py                        # Setup script for packaging
└── requirements.txt                # Python dependencies
`

`
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
├── scripts/
│   ├── install_falafel.sh          # Installation script
│   └── start_gunicorn.sh           # Script to start Gunicorn
|
└──requirements.txt
`

This Structure is used for Manual Installation.

In the context of Docker Container, the `setup.py` script and the bash scripts (in `script/*`) are redundant/irrelevant



