import os
from transform import Normilizer


models_path = "../load/store"

for file in os.listdir(models_path):
    if file.endswith(".obj"):
        file_path = os.path.join(models_path, file)
        
        transform_model = Normilizer(file_path)
        
        transform_model.move_to_zero()
        transform_model.normilize_to_unit_radius()
        
        transform_model.export_obj(file_path)