import bpy
import math
import mathutils
import pandas as pd


class Converter:
    
    def __init__(self, file_path):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.wm.obj_import(filepath=file_path)

            
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
