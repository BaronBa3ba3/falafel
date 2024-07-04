#!/bin/bash

# Define variables
SERVICE_NAME=falafel
SERVICE_FILE=/etc/systemd/system/${SERVICE_NAME}.service
USERNAME=$(whoami)
WORKING_DIRECTORY=/path/to/your/project
START_SCRIPT=${WORKING_DIRECTORY}/start_gunicorn.sh

# Create the start script if it doesn't exist
if [ ! -f "$START_SCRIPT" ]; then
    echo "Creating start script at $START_SCRIPT"
    cat << EOF > $START_SCRIPT
#!/bin/bash
# Set the working directory to the location of this script
cd ${WORKING_DIRECTORY}
# Activate virtual environment if you have one
source /path/to/your/venv/bin/activate
# Start Gunicorn
exec gunicorn --workers 3 --bind 0.0.0.0:8000 wsgi:create_app()
EOF
    chmod +x $START_SCRIPT
else
    echo "Start script already exists at $START_SCRIPT"
fi

# Create the service file
echo "Creating service file at $SERVICE_FILE"
sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Gunicorn instance to serve my Flask application
After=network.target

[Service]
User=$USERNAME
Group=www-data
WorkingDirectory=$WORKING_DIRECTORY
ExecStart=$START_SCRIPT

[Install]
WantedBy=multi-user.target
EOF

# Reload Systemd, start and enable the service
echo "Reloading Systemd, starting and enabling the service"
sudo systemctl daemon-reload
sudo systemctl start $SERVICE_NAME
sudo systemctl enable $SERVICE_NAME

echo "Service installed and started successfully."
