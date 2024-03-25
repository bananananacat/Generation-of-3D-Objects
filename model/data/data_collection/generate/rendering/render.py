import bpy
import random
from mathutils import Vector


class Render:

    def __init__(self, file_path):
        bpy.ops.wm.obj_import(filepath=file_path)


    def generate_random_camera_location(self):
        min_coord = -3
        max_coord = 3
        a, b, c = 0, 0, 0
        
        while (a <= 1 and a >= -1):
            a = random.uniform(min_coord, max_coord)
        while (b <= 1 and b >= -1):
            b = random.uniform(min_coord, max_coord)
        while (c <= 1 and c >= -1):
            c = random.uniform(min_coord, max_coord)
        
        camera_location = (a, b, c)
        return camera_location


    def render_from_view(self, camera_location, target_location, output_path, name, i, image_format='PNG', resolution=(1920, 1080)):
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        
        bpy.ops.object.camera_add(location=camera_location)
        camera = bpy.context.object
        
        target_location = Vector(target_location)
        camera_location = Vector(camera_location)

        direction = camera_location - target_location
        direction.normalize()

        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = direction.to_track_quat('Z', 'Y')

        bpy.context.scene.camera = camera
        bpy.ops.object.empty_add(location=target_location)
        target = bpy.context.object
        bpy.ops.render.render(write_still=True)
        
        bpy.data.images['Render Result'].save_render(filepath=output_path + "/" + f"{name}_{i}.png") # адекватное название рендеров с разных ракурсов
        bpy.data.objects.remove(camera)
        #bpy.data.objects.remove(target) - мб не понадобится, тк в creating_dataset удаляем все кроме света каждую итерацию
    
    
    