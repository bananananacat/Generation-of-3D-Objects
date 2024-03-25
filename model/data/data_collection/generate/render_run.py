import bpy
import os
from mathutils import Vector

from rendering import Render
from rendering import *

models_folder = "../load/store"
output_path = "./pics"

clear_area()
add_lights(get_cube_points())
   
for file_name in sorted(os.listdir(models_folder)):
    if file_name.endswith(".obj"):
        model_path = os.path.join(models_folder, file_name)
        
        render_object = Render(model_path)
        
        name = file_name[:-4]
        
        target_location = bpy.context.object.location
        for i in range(10):
            camera_location = render_object.generate_random_camera_location()
            render_object.render_from_view(camera_location, target_location, output_path, name, i, image_format='PNG', resolution=(1920, 1080))
        clear_area_without_lights()