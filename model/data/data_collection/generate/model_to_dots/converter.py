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

    
# example 

file_path = 'land-rover-range-rover-velar-2018-1.snapshot.5/Land-Rover_Range_Rover_Velar_First_edition_HQinterior_2018.obj'
export_path = 'first_export.obj'
csv_path = 'coords.csv'  

if __name__ == '__main__':
    convert_car = Converter(file_path)
    convert_car.move_objects(convert_car.find_min_point())
    convert_car.make_points_output(csv_path)
    
