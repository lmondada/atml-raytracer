import torch
import torch.nn.functional as F

from ..utils.constants import *
from ..geometry import Primitive, Collider


class Plane(Primitive):
    def __init__(
        self,
        center,
        material,
        width,
        height,
        u_axis,
        v_axis,
        max_ray_depth=5,
        shadow=True,
    ):
        super().__init__(center, material, max_ray_depth, shadow=shadow)
        self.collider_list += [
            Plane_Collider(
                assigned_primitive=self,
                center=center,
                u_axis=u_axis,
                v_axis=v_axis,
                w=width / 2,
                h=height / 2,
            )
        ]
        self.width = torch.tensor(width)
        self.height = torch.tensor(height)
        self.bounded_sphere_radius = torch.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)

    def get_uv(self, hit):
        return hit.collider.get_uv(hit)


class Plane_Collider(Collider):
    def __init__(self, u_axis, v_axis, w, h, uv_shift=(0.0, 0.0), **kwargs):
        super().__init__(**kwargs)
        self.u_axis = torch.as_tensor(u_axis)
        self.v_axis = torch.as_tensor(v_axis)
        self.normal = F.normalize(self.u_axis.cross(self.v_axis), dim=0)

        self.w = torch.tensor(w)
        self.h = torch.tensor(h)
        self.uv_shift = uv_shift
        self.inverse_basis_matrix = torch.tensor(
            [
                [self.u_axis[...,0], self.v_axis[...,0], self.normal[...,0]],
                [self.u_axis[...,1], self.v_axis[...,1], self.normal[...,1]],
                [self.u_axis[...,2], self.v_axis[...,2], self.normal[...,2]],
            ]
        )
        self.basis_matrix = self.inverse_basis_matrix.T

    def intersect(self, O, D):
        N = self.normal

        NdotD = N.inner(D)
        NdotD = torch.where(NdotD == 0.0, NdotD + 0.0001, NdotD)  # avoid zero division

        NdotC_O = N.inner(self.center - O)
        d = D * (NdotC_O / NdotD).reshape(-1, 1)
        M = O + d  # intersection point
        dis = torch.norm(d, dim=1)

        M_C = M - self.center

        # plane basis coordinates
        u = self.u_axis.inner(M_C)
        v = self.v_axis.inner(M_C)

        hit_inside = (
            (torch.abs(u) <= self.w) & (torch.abs(v) <= self.h) & (NdotC_O * NdotD > 0)
        )
        hit_UPWARDS = NdotD < 0
        hit_UPDOWN = ~hit_UPWARDS

        pred1 = hit_inside & hit_UPWARDS
        pred2 = hit_inside & hit_UPDOWN

        ret1 = torch.where(pred1 | pred2, dis, torch.full(dis.shape, FARAWAY))
        ret2 = torch.where(pred2, UPDOWN, torch.full(dis.shape, FARAWAY))
        ret2 = torch.where(pred1, UPWARDS, ret2)
        return (ret1, ret2)

#     def rotate(self, M, center):
#         self.u_axis = self.u_axis.matmul(M)
#         self.v_axis = self.v_axis.matmul(M)
#         self.normal = self.normal.matmul(M)
#         self.center = center + (self.center - center).matmul(M)

    def get_uv(self, hit):
        M_C = hit.point - self.center
        u = (self.u_axis.dot(M_C) / self.w + 1) / 2 + self.uv_shift[0]
        v = (self.v_axis.dot(M_C) / self.h + 1) / 2 + self.uv_shift[1]
        return u, v

    def get_Normal(self, hit):
        return self.normal
