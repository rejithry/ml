import pyautogui

import time

#time.sleep(5)

#pyautogui.keyDown('w')

#time.sleep(3)

from key_poller import KeyPoller

if __name__ == '__main__':
  with KeyPoller() as keyPoller:
    while True:
        c = keyPoller.poll()
        if not c is None:
            if c == "c":
                break
            print c