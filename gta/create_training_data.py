# create_training_data.py

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

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D,S, No action] boolean values.
    '''
    output = [0,0,0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'S' in keys:
        output[3] = 1
    else:
        output[4] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    
    t = threading.Thread(target=listen)
    
    t.start()
    t.join


    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = screenshot(0,40,640,480)
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            print keys
            output = keys_to_output(keys)
            training_data.append([screen,output])
            
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name,training_data)

        keys = key_check()
        print keys
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()