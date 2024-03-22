import rendering_func
def generate_random_camera_location():
    min_coord = -30
    max_coord = 30
    camera_location = (
        random.uniform(min_coord, max_coord),
        random.uniform(min_coord, max_coord),
        random.uniform(min_coord, max_coord)
    )
    return camera_location
models_folder = "/home/banana_cat"
output_path = "/home/banana_cat/rended_pics"
#output_path = "//wsl.localhost/Ubuntu-22.04/home/banana_cat/rended_pics"
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

for file_name in os.listdir(models_folder):
    if file_name.endswith(".obj"):
        model_path = os.path.join(models_folder, file_name)
        bpy.ops.wm.obj_import(filepath=model_path)
        # Среднее положение вершин модели
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        target_location = bpy.context.object.location
        camera_location = generate_random_camera_location()
        #for i in range(10):
        render_from_view(camera_location, target_location, output_path, image_format='PNG', resolution=(1920, 1080))

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.scene.objects.active = bpy.data.objects[file_name[:-4]]
        bpy.data.objects[file_name[:-4]].select = True
        bpy.ops.object.delete()
