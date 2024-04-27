import bpy
import math
import mathutils
import pandas as pd


class Converter:
    
    def __init__(self, file_path):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.wm.obj_import(filepath=file_path)


    def find_min_point(self):
        min_point = [math.inf, math.inf, math.inf]
        
        for key in bpy.data.objects.keys():
            obj = bpy.data.objects[key]
            coords = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
            for coord in coords:
                min_point[0] = min(coord[0], min_point[0])
                min_point[1] = min(coord[1], min_point[1])
                min_point[2] = min(coord[2], min_point[2])

        return mathutils.Vector(min_point)


    def find_max_point(self):
        max_point = [-math.inf, -math.inf, -math.inf]

        for key in bpy.data.objects.keys():
            obj = bpy.data.objects[key]
            coords = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
            for coord in coords:
                max_point[0] = max(coord[0], max_point[0])
                max_point[1] = max(coord[1], max_point[1])
                max_point[2] = max(coord[2], max_point[2])

        return mathutils.Vector(max_point)


    def find_max_by_euclidian(self):
        radius = 0
        for key in bpy.data.objects.keys():
            obj = bpy.data.objects[key]
            coords = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
            for coord in coords:
                radius = max(((coord[0] ** 2 + coord[1] ** 2 + coord[2] **2) ** 0.5), radius)
    
        return radius


    def erase_extra_objects(self, erase_list):
        for mesh in erase_list:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[mesh].select_set(True)    
            bpy.ops.object.delete()


    def move_objects(self, delta):
        for key in bpy.data.objects.keys():
            obj = bpy.data.objects[key]
            obj.location = obj.location - delta


    def scale_objects(self, scale):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        for key in bpy.data.objects.keys():
            obj = bpy.data.objects[key]
            obj.scale = scale


    def make_points_output(self, out_path):
        output = {'x': [], 'y': [], 'z': []}
        for key in bpy.data.objects.keys():
            obj = bpy.data.objects[key]
            coords = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
            for coord in coords:
                output['x'].append(coord[0])
                output['y'].append(coord[1])
                output['z'].append(coord[2])
        df = pd.DataFrame(output)
        df = df.drop_duplicates()
        df.to_csv(out_path)
