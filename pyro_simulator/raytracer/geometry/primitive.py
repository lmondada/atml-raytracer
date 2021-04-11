import torch
import torch.nn.functional as F

from math import pi


class Primitive:
    def __init__(self, center, material, max_ray_depth=1, shadow=True):
        self.center = torch.as_tensor(center)
        self.material = material
        self.material.assigned_primitive = self
        self.shadow = shadow
        self.collider_list = []
        self.max_ray_depth = max_ray_depth

#     def rotate(self, θ, u):
# 
#         u = F.normalize(u)
#         θ = θ / 180 * pi
#         cosθ = torch.cos(θ)
#         sinθ = torch.sqrt(1 - cosθ ** 2) * torch.sign(θ)
# 
#         # rotation matrix along u axis
#         M = torch.tensor(
#             [
#                 [
#                     cosθ + u.x * u.x * (1 - cosθ),
#                     u.x * u.y * (1 - cosθ) - u.z * sinθ,
#                     u.x * u.z * (1 - cosθ) + u.y * sinθ,
#                 ],
#                 [
#                     u.y * u.x * (1 - cosθ) + u.z * sinθ,
#                     cosθ + u.y ** 2 * (1 - cosθ),
#                     u.y * u.z * (1 - cosθ) - u.x * sinθ,
#                 ],
#                 [
#                     u.z * u.x * (1 - cosθ) - u.y * sinθ,
#                     u.z * u.y * (1 - cosθ) + u.x * sinθ,
#                     cosθ + u.z * u.z * (1 - cosθ),
#                 ],
#             ]
#         )
#         for c in self.collider_list:
#             c.rotate(M, self.center)
