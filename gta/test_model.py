# test_model.py

import numpy as np
from image_grab import screenshot
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
import tensorflow as tf

import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09

import numpy as np
from image_grab import screenshot
import cv2
import time
import os

import zmq
import time
import threading

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5555')

buffer = []

def key_check():
  global buffer
  all_presses_sofar=''.join(buffer)
  buffer = []
  return all_presses_sofar

def listen():
  global buffer
  while True:
      message = socket.recv()
      buffer.append(message)
      socket.send('A')

def straight():
##    if random.randrange(4) == 2:
##        ReleaseKey(W)
##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)

def reverse():
    PressKey(S)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)

def donothing():
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(W)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = screenshot(0,40,640, 480)
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            prediction = model.predict([screen.reshape(160,120,1)])[0]
            print(prediction)

            turn_thresh = .30
            fwd_thresh = 0.50

            p = np.argmax(prediction)

            if p == 1 and prediction[1] > fwd_thresh:
                straight()
                time.sleep(3)
            elif p == 0 and prediction[0] > turn_thresh:
                left()
                time.sleep(1)
            elif p == 2 and prediction[2] > turn_thresh:
                right()
                time.sleep(1)
            elif p == 3 and prediction[3] > turn_thresh:
                reverse()
                time.sleep(1)
            else:
                donothing()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       









