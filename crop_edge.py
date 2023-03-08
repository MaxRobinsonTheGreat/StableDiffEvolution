import numpy as np
from PIL import Image
from util import toPil

image_path = 'outpaints/ai_museum_left_2/full.png'
edge_size = 912
right = False

im = np.array(Image.open(image_path))
if right: 
    im = im[:,-edge_size:]
else:
    im = im[:,:edge_size]
toPil(im).save('crop.png')