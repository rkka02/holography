import slmpy
import time

def update(image):
    slm = slmpy.SLMdisplay(monitor=0)
    resX, resY = slm.getSize()
    slm.updateArray(image)
    time.sleep(10)