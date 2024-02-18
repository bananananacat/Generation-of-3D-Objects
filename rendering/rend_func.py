import bpy

def render_from_view(camera_location, target_location, output_path, image_format='PNG', resolution=(1920, 1080)):
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    # Камера
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object

    bpy.context.scene.camera = camera
    bpy.ops.object.empty_add(location=target_location)
    target = bpy.context.object

    bpy.ops.render.render(write_still=True)
    #bpy.data.images['Render Result'].save_render(filepath=output_path, file_format=image_format)


