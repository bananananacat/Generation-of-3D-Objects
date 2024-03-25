from .model_transformer import Transormer


class Normilizer(Transormer):
    
    def move_to_zero(self):
        min_point = self.find_min_point()
        max_point = self.find_max_point()
        self.move_objects(min_point + (max_point - min_point) / 2)


    def normilize_to_unit_radius(self):
        r = self.find_max_by_euclidian()
        self.scale_objects((1/r, 1/r, 1/r))
