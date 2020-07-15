
# coding: utf-8

# In[ ]:


# Setup the connection between our model and the simulator in order to get the car to  drive basied on the
# steering angles provided by the trained model(bidirectional client server)
# communication
# 1). so we need to install python-scoketio.
# ************************************************************
# IMPORTANT NOTE:
# Generally we will install the socketio module directly pip, and socketio.Server() This package is only 
# available in python-socketio. socketio and python-socketio are two different modules.
# pip uninstall socketio
# then 
# pip install python-socketio
# ************************************************************

# 2). We need to initialize a python web application and to do so we need to install Flask as well.
# Flask: is a python micro framework that's used to build web apps.


# In[ ]:


import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


# In[ ]:


# Initialize our web server to perform real time communication between client and server.
# server will keep listening to event when client creates a single connection to a web socket server.
sio = socketio.Server()

# Create our python web applicaiton.
# __name__ is a special variable will ends up having the value of '__main__'
app = Flask(__name__)

# Limit the speed of the car.
speed_limit = 10


# In[ ]:


def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


# In[ ]:


# Sends data (steering angle and throttle) to our simulator.
def send_control(steering_angle, throttle):
    # Emit the given data to our simulator.
    sio.emit('steer', data = { 'steering_angle': steering_angle.__str__(),
                               'throttle': throttle.__str__()
    })


# In[ ]:


# Register a special event - telemetry(sessionID, data)
# The simulator will send us back data which contains image of the current frame (car current position)
# then we will run this image into our model to give us the predicted steering angle.
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


# In[ ]:


# On Connect event (sessionID, environment)
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


# In[ ]:


# Check if we executing the script then we want run our applicaiton.
if __name__ == '__main__':
    # Load our model.
    model = load_model('model.h5')
    # we need a middleware to dispatch traffic to a socketio (combine our sio with our app).
    app = socketio.Middleware(sio, app)
    # make our web server send any requests that made by the client to our web application itself.
    # NOTE: eventlet.listen: will open listen socket. 
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

