import zmq
import time
import threading

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5555')

buffer = []



def listen():
  while True:
      message = socket.recv()
      buffer.append(message)
      socket.send('A')


t = threading.Thread(target=listen)

t.start()
t.join


while True:
  time.sleep(3)
  print buffer
  buffer = []