import trimesh


filename = "duck.glb"
mesh = trimesh.load(filename)
for shape_name, data in mesh.geometry.items():
    print("vertices:",data.vertices)
