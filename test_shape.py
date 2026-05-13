import tifffile as tiff
import sys
import glob
import numpy as np

files = glob.glob('data/raw/*/*/*.TIF') + glob.glob('data/raw/*/*/*.tif')
if files:
    img = tiff.imread(files[0])
    print("Shape:", img.shape)
    # Check what axes means
    with tiff.TiffFile(files[0]) as tif:
        print("Axes:", tif.series[0].axes)
else:
    print("No TIF found")
