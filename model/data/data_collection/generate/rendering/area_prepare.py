import bpy


def clear_area():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def clear_area_without_lights():
        for obj in bpy.data.objects: # после рендера с 10 ракурсов удаляем все кроме света, камеру удаляем на каждой итерациии и создаем заново
            if obj.type != 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)


def get_cube_points():
    result = []
    list = [-1, 1]

    for i in list:
        for j in list:
            for k in list:
                result.append((i, j, k))
    
    return result
    

def add_lights(locations):
    for location in locations:
        bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=location)
