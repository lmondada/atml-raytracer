from abc import abstractmethod
import torch

class Collider:
    def __init__(self, assigned_primitive, center):
        self.assigned_primitive = assigned_primitive
        self.center = torch.as_tensor(center)

    @abstractmethod
    def intersect(self, O, D):
        pass

    @abstractmethod
    def get_Normal(self, hit):
        pass
