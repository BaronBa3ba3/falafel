#!/bin/bash

# Variables
SERVICE_NAME=falafel
SERVICE_FILE=/etc/systemd/system/${SERVICE_NAME}.service
USERNAME=$(whoami)
WORKING_DIRECTORY=/path/to/your/application
START_SCRIPT=${WORKING_DIRECTORY}/start_gunicorn.sh
VENV_DIR=${WORKING_DIRECTORY}/venv

# Function to create the start script
create_start_script() {
    if [ ! -f "$START_SCRIPT" ]; then
        echo "Creating start script at $START_SCRIPT"
        cat << EOF > $START_SCRIPT
#!/bin/bash
# Set the working directory to the location of this script
cd ${WORKING_DIRECTORY}
# Activate virtual environment if you have one
source ${VENV_DIR}/bin/activate
# Start Gunicorn
exec gunicorn --workers 3 --bind 0.0.0.0:8000 falafel.wsgi:create_app()
EOF
        chmod +x $START_SCRIPT
    else
        echo "Start script already exists at $START_SCRIPT"
    fi
}

# Function to create the service file
create_service_file() {
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
}

# Update and install dependencies
echo "Updating package lists..."
sudo apt update

echo "Installing required packages..."
sudo apt install -y python3 python3-pip python3-venv

# Set up the application
echo "Setting up the application..."
cd $WORKING_DIRECTORY

echo "Creating virtual environment..."
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing the application as a package..."
pip install .

# Create start script and service file
create_start_script
create_service_file

# Reload systemd, start, and enable the service
echo "Reloading Systemd..."
sudo systemctl daemon-reload

echo "Starting the service..."
sudo systemctl start $SERVICE_NAME

echo "Enabling the service to start on boot..."
sudo systemctl enable $SERVICE_NAME

echo "Service installed and started successfully."













# Install Dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv


# Navigate to your application directory
cd /path/to/your/application


# Create and Activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Python Dependencies
pip install -r requirements.txt

# Install the Application as a Package
pip install .

