import torch
import numpy as np


class DotsLoader:

    def normilize_dots(self, dots):
        dots -= np.min(dots)
        dots /= np.max(dots)
        return dots

    def open_dots(self, dots_path):
        dots = np.load(dots_path)
        dots = self.normilize_dots(dots)
        dots = torch.from_numpy(dots).float()

        return dots
