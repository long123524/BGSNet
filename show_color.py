import numpy as np
import os
from datasets import RS_ST as RS
from skimage import io

dir = r'D:\label2'
save_path = r'D:\L2_newRGB'

for filename in os.listdir(dir):
    input_path = os.path.join(dir, filename)
    img = io.imread(input_path)
    io.imsave(os.path.join(save_path,filename[:-4]+'.tif'), RS.Index2Color(img))





