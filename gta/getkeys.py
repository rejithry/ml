from AppKit import NSApplication, NSApp
from Foundation import NSObject, NSLog
from Cocoa import NSEvent, NSKeyDownMask
from PyObjCTools import AppHelper
import zmq


import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5555')

keyMap = {126 : 'W', 124 : 'D', 123 : 'A', 125 : 'S', 0 : 'A', 13 : 'W', 1 : 'S', 2 : 'D', 17 : 'T'}


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        mask = NSKeyDownMask
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, handler)


def handler(event):
    try:
        key = event.keyCode()
        if key in keyMap:
            socket.send(keyMap[key])
            socket.recv()

    except KeyboardInterrupt:
        AppHelper.stopEventLoop()

def start_keylogger():
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    NSApp().setDelegate_(delegate)
    AppHelper.runEventLoop()
    
if __name__ == '__main__':
    start_keylogger()