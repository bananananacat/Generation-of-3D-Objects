from rend_func.py import *

import bpy
import os
import random
from mathutils import Vector

def generate_random_camera_location():
    min_coord = -10
    max_coord = 10
    camera_location = (
        random.uniform(min_coord, max_coord),
        random.uniform(min_coord, max_coord),
        random.uniform(min_coord, max_coord)
    )
    return camera_location

bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

models_folder = "C:\\Users\\Mark\\Documents\\3d_models"
output_path = "./"
#output_path = "Ubuntu-22.04/home/banana_cat/rended_pics"

for file_name in os.listdir(models_folder):
    if file_name.endswith(".obj"):
        model_path = os.path.join(models_folder, file_name)
        bpy.ops.wm.obj_import(filepath=model_path)
        # Среднее положение вершин модели
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        target_location = bpy.context.object.location
        print(target_location)
        camera_location = generate_random_camera_location()
        render_from_view(camera_location, target_location, output_path, image_format='PNG', resolution=(1920, 1080))
