import cv2
import numpy as np
import time
import pickle
from tensorflow import keras
import tensorflow as tf

with open('model_pickle','rb') as f:
    model = pickle.load(f)
classes = ['paper', 'rock', 'scissors']

cap = cv2.VideoCapture('your_video.mp4')

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    image = frame
    frame = cv2.resize(frame, (32, 32))
    frame = tf.convert_to_tensor(frame, dtype=tf.float32)
    frame = tf.expand_dims(frame, axis=0)
    frame = frame/255
    soln = model.predict(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = frame.numpy()
    label = classes[np.argmax(soln)]
    image = cv2.putText(image, label, (50,50) , font, 
                   1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('ML', image) 

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()