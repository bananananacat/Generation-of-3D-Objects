import os


def convert_photos_to_3d_model(upload_path):
    model_path = os.path.join(upload_path, 'result.obj')
    success = True
    return success, model_path
