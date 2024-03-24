from ..model_to_dots.converter import Converter

temp = Converter("/home/banana_cat/3d_models/beast.obj")
r = temp.find_max_by_euclidian()
temp.move_objects(temp.find_min_point() + (temp.find_max_point() - temp.find_min_point()) / 2)
temp.export_obj("/home/banana_cat/beast2.obj")

temp = Converter("/home/banana_cat/beast2.obj")
r = temp.find_max_by_euclidian()
temp.scale_objects((1/r, 1/r, 1/r))
temp.export_obj("/home/banana_cat/beast2.obj")

temp = Converter("/home/banana_cat/beast2.obj")
temp.move_objects(temp.find_min_point() + (temp.find_max_point() - temp.find_min_point()) / 2)
temp.export_obj("/home/banana_cat/beast2.obj")
