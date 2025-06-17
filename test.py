import slmpy
import numpy as np
import hologram
from PIL import Image
import time
import scipy
from scipy.interpolate import interp1d

# Loads the SLM lookup table from the .mat file and computes phaseVal and SLMDigit.
def SLM_LUT():
    # Load the MAT file containing 'totalSum'
    mat_data = scipy.io.loadmat('CGH_tutorial/HW_58_pol_60_phase_correct.mat')
    totalSum = mat_data['totalSum']
    # Assume totalSum is a column/row vector; flip it vertically
    flipphase = np.flipud(totalSum).flatten()
    # Create ydata from 255 to 0 in 256 steps
    ydata = np.linspace(255, 0, 256)
    # Get unique values (sorted) and the indices
    phaseVal, ia = np.unique(flipphase, return_index=True)
    SLMDigit = ydata[ia]
    # Center phaseVal to be in [-pi, pi]
    phaseCenter = (np.max(phaseVal) + np.min(phaseVal)) / 2
    phaseVal = phaseVal - phaseCenter
    return phaseVal, SLMDigit


if __name__=='__main__':
    try:
        slm = slmpy.SLMdisplay(monitor=0)
        resX, resY = slm.getSize()

        image1 = Image.open('test.png').convert('L')
        images = [image1]
        zs = [0]
        patterns = []
        for image in images:
            image = np.array(image)
            image = hologram.scale_and_pad(image, (resY, resX), 0)
            # image = np.flipud(image)
            image = np.fliplr(image)
            patterns.append(image)
        
        # cgh = hologram.calculate_hologram(patterns, zs, resX, resY)
        slm.updateArray(image)
        time.sleep(600)
        slm.close()
    except KeyboardInterrupt:
        slm.close()