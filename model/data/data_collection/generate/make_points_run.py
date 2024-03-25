import os
from model_to_dots import Converter


models_path = "../load/store"
out_path = "./points"

for file in os.listdir(models_path):
    if file.endswith(".obj"):
        file_path = os.path.join(models_path, file)
        
        filename = file.split('.')[0]
        current_out_path = os.path.join(out_path, filename + '.csv')
                
        convert_model = Converter(file_path)
        convert_model.make_points_output(current_out_path)
