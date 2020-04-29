import png
import numpy as np
from PIL import Image

r = png.Reader(filename='cartonPNG.png').asRGB()
image_2d = np.vstack(map(np.uint8, r[2]))
print(image_2d)

r = png.Reader(filename='cartonPNG.png').read_flat()
image_2d = np.array(r[2], np.int8)
print(image_2d)


image = Image.open("cartonPNG.png")
img_data = np.array(list(image.getdata()), np.uint8)
print(img_data)

#image_3d = np.reshape(image_2d, (r[0]*r[1], 3))
#print(image_3d)

"""
texture_data = png.Reader(filename='test.png').read_flat()

print(texture_data[0])
print(texture_data[1])
print(np.array(texture_data[2]).shape)
"""

