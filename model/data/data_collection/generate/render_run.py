import bpy
import os
from mathutils import Vector

from rendering import Render
from rendering import *


models_folder = "../load/store"
output_path = "./pics"

clear_area_without_lights()
add_lights(get_cube_points())
   
for file_name in os.listdir(models_folder):
    if file_name.endswith(".obj"):
        model_path = os.path.join(models_folder, file_name)
        name = file_name[:-4]
        
        render_object = Render(model_path)        
        target_location = bpy.context.object.location
        
        out_dir = os.path.join(output_path, name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for i in range(10):
            camera_location = render_object.generate_random_camera_location()
            render_object.render_from_view(camera_location, target_location, out_dir, name, i, image_format='PNG', resolution=(1920, 1080))
        clear_area_without_lights()
