import bpy
import os
import random
from mathutils import Vector

def render_from_view(camera_location, target_location, output_path, image_format='PNG', resolution=(1920, 1080)):
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object

    # Определяем вектор направления от камеры до объекта
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
    
    bpy.data.images['Render Result'].save_render(filepath=output_path + "/5.png")
    bpy.data.objects.remove(camera)
    bpy.data.objects.remove(target)


