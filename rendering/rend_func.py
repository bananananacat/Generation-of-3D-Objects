import bpy
# для каждой картинки 10 раз генерируем рандомное camera_location и поворот и 10 раз делаем эту функцию
# за target_location можно брать центр, т е (0,0,0), если объекты будут далеко от центра, то надо написать
# еще одну функцию, которая берет центр/ как-то двигает параллельным переносом центр модели или камеру 
# хотя самым простым решением в данной ситуации будет установка камеры так , чтобы по одной координате она 
# была не сильно далеко от центра модели 
def render_from_view(camera_location, target_location, output_path, image_format='OBJ', resolution=(1920, 1080)):
    
    # разрешение рендера
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    # камера
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.rotation_euler = (1.0472, 0, 0.7854)  # тут поворачиваем если надо
    # точка обзора
    bpy.context.scene.camera = camera
    bpy.ops.object.empty_add(location=target_location)
    target = bpy.context.object

    # Рендеринг
    bpy.ops.render.render(write_still=True)

    # Сохранение
    bpy.data.images['Render Result'].save_render(filepath=output_path, file_format=image_format)

    # Удаление камеры и точки обзора
    bpy.data.objects.remove(camera)
    bpy.data.objects.remove(target)
