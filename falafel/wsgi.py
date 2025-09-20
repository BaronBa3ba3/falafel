from falafel.app import app, socketio



if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)      # For Flask 
    socketio.run(app)                       # For Flask-SocketIO

