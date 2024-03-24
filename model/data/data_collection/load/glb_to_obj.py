import bpy


def import_glb(file_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=file_path)


def export_obj(export_path):
    bpy.ops.wm.obj_export(filepath=export_path)