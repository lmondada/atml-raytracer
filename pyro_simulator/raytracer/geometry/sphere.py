import torch
from math import pi

from ..utils.constants import *
from ..geometry import Primitive, Collider

class Sphere(Primitive):
    def __init__(
        self, center, material, radius, max_ray_depth=5, shadow=True
    ):
        super().__init__(center, material, max_ray_depth, shadow=shadow)
        self.collider_list += [
            Sphere_Collider(assigned_primitive=self, center=center, radius=radius)
        ]
        self.bounded_sphere_radius = radius

    def get_uv(self, hit):
        return hit.collider.get_uv(hit)


class Sphere_Collider(Collider):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def intersect(self, O, D):

        b = 2 * torch.sum(D * (O - self.center), dim=1)
        c = (
            self.center.norm()**2
            + O.norm(dim=-1)**2
            - 2 * self.center.inner(O)
            - (self.radius * self.radius)
        )
        disc = (b ** 2) - (4 * c)
        sq = torch.sqrt(torch.maximum(torch.tensor(0.), disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = torch.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        M = O + D * h.reshape(-1, 1)
        NdotD = torch.sum(
            ((M - self.center) * (1.0 / self.radius)) * D,
            dim=-1
        )

        pred1 = (disc > 0) & (h > 0) & (NdotD > 0)
        pred2 = (disc > 0) & (h > 0) & (NdotD < 0)
        pred3 = True

        # return an array with hit distance and the hit orientation
        ret1 = torch.where(pred1 | pred2, h, torch.full(h.shape, FARAWAY))
        ret2 = torch.where(pred2, UPDOWN, torch.full(h.shape, FARAWAY))
        ret2 = torch.where(pred1, UPWARDS, ret2)
        return (ret1, ret2)
#         return np.select(
#             [pred1, pred2, pred3],
#             [[h, np.tile(UPDOWN, h.shape)], [h, np.tile(UPWARDS, h.shape)], FARAWAY],
#         )

    def get_Normal(self, hit):
        # M = intersection point
        return (hit.point - self.center) * (1.0 / self.radius)

    def get_uv(self, hit):
        M_C = (hit.point - self.center) / self.radius
        phi = torch.arctan2(M_C.z, M_C.x)
        theta = torch.arcsin(M_C.y)
        u = (phi + pi) / (2 * pi)
        v = (theta + pi / 2) / pi
        return u, v
