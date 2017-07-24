#!/usr/bin/env python
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Usage:
#
# Goto System Preferences > Security & Privacy > Accessibility and add
# Python to apps allowed to control your computer. If it is not in the
# list, the easiest is to run this file first and it should appear.

from AppKit import NSApplication, NSApp
from Foundation import NSObject, NSLog
from Cocoa import NSEvent, NSKeyDownMask
from PyObjCTools import AppHelper

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        mask = NSKeyDownMask
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, handler)

def handler(event):
    try:
        NSLog(u"%@", event)
        print event
    except KeyboardInterrupt:
        AppHelper.stopEventLoop()

def main():
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    NSApp().setDelegate_(delegate)
    AppHelper.runEventLoop()
    
if __name__ == '__main__':
    main()
