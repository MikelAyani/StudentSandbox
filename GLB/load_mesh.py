from PIL import Image #Pillow library
from PIL import ImageOps
import trimesh
 
glb_filename = "GLB/duck.glb"
mesh_duck = trimesh.load(glb_filename)

for shape_name, data in mesh_duck.geometry.items():
    image_texture = data.visual.material.baseColorTexture
    uv_coordinates = data.visual.uv

image_texture.save("GLB/reee.png","PNG")

