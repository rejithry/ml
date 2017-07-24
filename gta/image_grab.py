import Quartz
import Quartz.CoreGraphics as CG
import numpy as np

# This is much faster than ImageGrab
def screenshot(x, y, w, h):
    region = CG.CGRectMake(x, y, w, h)

    # Create screenshot as CGImage
    image = CG.CGWindowListCreateImage(
        region,
        CG.kCGWindowListOptionOnScreenOnly,
        CG.kCGNullWindowID,
        CG.kCGWindowImageDefault)

    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    bytesperrow = CG.CGImageGetBytesPerRow(image)

    pixeldata = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
    image =  np.frombuffer(pixeldata, dtype=np.uint8)
    image = image.reshape((height, bytesperrow//4, 4))
    return image[:,:width,:]

