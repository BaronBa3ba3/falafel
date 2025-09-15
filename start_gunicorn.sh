#!/bin/bash

# Set the working directory to the location of this script
cd "$(dirname "$0")"

# Activate virtual environment if you have one
source venv/bin/activate

# Start Gunicorn
exec gunicorn --workers 3 --bind 0.0.0.0:8000 myapp.wsgi:app


# To make scritp executable : chmod +x start_gunicorn.sh
