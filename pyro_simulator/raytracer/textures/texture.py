from ..utils.constants import *
from abc import abstractmethod


class texture:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_color(self, hit):
        pass


class solid_color(texture):
    def __init__(self, color):
        self.color = color

    def get_color(self, hit):
        return self.color
