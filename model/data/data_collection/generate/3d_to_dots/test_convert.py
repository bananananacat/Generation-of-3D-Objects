import bpy
import os
from math import radians
from bpy import context

file_loc = 'land-rover-range-rover-velar-2018-1.snapshot.5/Land-Rover_Range_Rover_Velar_First_edition_HQinterior_2018.obj' 
imported_object = bpy.ops.wm.obj_import(filepath=file_loc)
bpy.ops.wm.obj_import(filepath=file_loc)
obj_object = bpy.context.selected_objects[0]
print('Imported name: ', obj_object.name)

obj = context.active_object

coords = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
print (coords, len(coords))

