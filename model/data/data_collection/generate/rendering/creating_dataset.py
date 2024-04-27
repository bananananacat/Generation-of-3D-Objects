from rend_func import *

import bpy
import os
import random
from mathutils import Vector

def generate_random_camera_location():
    min_coord = -3
    max_coord = 3
    a = 0
    b = 0
    c = 0
    while (a <= 1 and a >= -1):
        a = random.uniform(min_coord, max_coord)
    while (b <= 1 and b >= -1):
        b = random.uniform(min_coord, max_coord)
    while (c <= 1 and c >= -1):
        c = random.uniform(min_coord, max_coord)
    camera_location = (
        a,
        b,
        c
    )
    return camera_location

bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

#models_folder = "/home/banana_cat/3d_models"
models_folder = "/home/banana_cat"
output_path = "/home/banana_cat/rended_pics"

list = [-1, 1]

for i in list:
    for j in list:
        for k in list:
            bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(i, j, k))
   
for file_name in os.listdir(models_folder):
    if file_name.endswith("cow2.obj"):
        model_path = os.path.join(models_folder, file_name)
        bpy.ops.wm.obj_import(filepath=model_path)
        name = file_name[:-4]
        target_location = bpy.context.object.location
        for i in range(10):
            camera_location = generate_random_camera_location()
            render_from_view(camera_location, target_location, output_path, name, i, image_format='PNG', resolution=(1920, 1080))
        for obj in bpy.data.objects:# после рендера с 10 ракурсов удаляем все кроме света, камеру удаляем на каждой итерациии и создаем заново
            if obj.type != 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)
