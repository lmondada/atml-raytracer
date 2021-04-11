from math import pi
from abc import abstractmethod

import torch
import torch.nn.functional as F
from pyro import distributions as distr


def random_in_unit_disk(*shape):
    r = torch.sqrt(distr.Uniform(0., 1.).sample(shape))
    phi = distr.Uniform(0., 2 * pi).sample(shape)
    return r * torch.cos(phi), r * torch.sin(phi)


def random_in_unit_sphere(*shape):
    # https://mathworld.wolfram.com/SpherePointPicking.html
    phi = distr.Uniform(0., 2 * pi).sample(shape)
    u = distr.Uniform(-1, 1).sample(shape)
    r = torch.sqrt(1 - u ** 2)
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), u], dim=-1)


class PDF:
    """Probability density function"""

    @abstractmethod
    def value(self, ray_dir):
        """get probability density function value at direction ray_dir"""
        pass

    @abstractmethod
    def generate(self):
        """generate random ray  directions according the probability density function"""
        pass


# class hemisphere_pdf(PDF):
#     """Probability density Function"""
# 
#     def __init__(self, shape, normal):
#         self.shape = shape
#         self.normal = normal
# 
#     def value(self, ray_dir):
#         val = 1.0 / (2.0 * np.pi)
#         assert val <= 1
#         assert val >= 0
#         return val
# 
#     def generate(self):
#         r = random_in_unit_sphere(self.shape)
#         return vec3.where(self.normal.dot(r) < 0.0, r * -1.0, r)
# 

class cosine_pdf(PDF):
    """Probability density Function"""

    def __init__(self, shape, normal):
        self.shape = torch.as_tensor(shape)
        self.normal = torch.as_tensor(normal)

    def value(self, ray_dir):
        val = torch.clip(torch.sum(ray_dir *self.normal, dim=1), 0.0, 1.0) / pi
        return val

    def generate(self):
        ax_w = self.normal
        a = torch.eye(1,3).repeat(len(ax_w), 1)
        a[ax_w[:, 0].abs() > 0.9] = torch.tensor([0., 1., 0.])
        ax_v = F.normalize(ax_w.cross(a), dim=-1)
        ax_u = ax_w.cross(ax_v)

        phi = distr.Uniform(0., 2 * pi).sample((self.shape,))
        r2 = distr.Uniform(0., 1.).sample((self.shape,))

        z = torch.sqrt(1 - r2)
        x = torch.cos(phi) * torch.sqrt(r2)
        y = torch.sin(phi) * torch.sqrt(r2)

        return ax_u * x.reshape(-1, 1) + ax_v * y.reshape(-1, 1) + ax_w * z.reshape(-1, 1)


class spherical_caps_pdf(PDF):
    """Probability density Function"""

    def __init__(self, shape, origin, importance_sampled_list):
        self.shape = shape
        self.origin = torch.as_tensor(origin)
        self.importance_sampled_list = importance_sampled_list
        self.l = len(importance_sampled_list)

    def value(self, ray_dir):
        PDF_value = 0.0
        for i in range(self.l):
            PDF_value += torch.where(
                torch.sum(ray_dir * self.ax_w_list[i], dim=-1) > self.cosθmax_list[i],
                # this was not a probability - has been changed!!
                torch.clip((1 - self.cosθmax_list[i]) * 2 * pi, 0, 1),
                torch.tensor(0.0, dtype=torch.float32),
            )
        PDF_value = PDF_value / self.l
        assert torch.max(PDF_value) <= 1
        assert torch.min(PDF_value) >= 0
        return PDF_value

    def generate(self):
        shape = self.shape
        origin = self.origin
        importance_sampled_list = self.importance_sampled_list
        l = self.l

        mask = distr.Categorical(probs=torch.ones(l)).sample((shape,))
        mask_list = [None] * l

        cosθmax_list = [None] * l
        ax_u_list = [None] * l
        ax_v_list = [None] * l
        ax_w_list = [None] * l

        for i in range(l):
            ax_w = F.normalize(importance_sampled_list[i].center - origin, dim=-1)
            ax_w_list[i] = ax_w
            a = torch.eye(1,3).repeat(len(ax_w), 1)
            a[ax_w[:, 0].abs() > 0.9] = torch.tensor([0., 1., 0.])
            ax_v_list[i] = F.normalize(ax_w_list[i].cross(a), dim=-1)
            ax_u_list[i] = ax_w_list[i].cross(ax_v_list[i])
            mask_list[i] = mask == i

            target_distance = torch.linalg.norm(
                importance_sampled_list[i].center - origin, dim=-1
            )

            cosθmax_list[i] = torch.sqrt(
                1
                - torch.clip(
                    importance_sampled_list[i].bounded_sphere_radius / target_distance,
                    0.0,
                    1.0,
                )
                ** 2
            )

        self.cosθmax_list = cosθmax_list
        self.ax_w_list = ax_w_list

        phi = distr.Uniform(0., 2 * pi).sample((shape,))
        r2 = distr.Uniform(0., 1.).sample((shape,))

        cosθmax = torch.zeros(*(cosθmax_list[0].shape))
        ax_w = torch.zeros(*(ax_w_list[0].shape))
        ax_v = torch.zeros(*(ax_v_list[0].shape))
        ax_u = torch.zeros(*(ax_u_list[0].shape))
        for i, mask in enumerate(mask_list):
            cosθmax[mask] = cosθmax_list[i][mask]
            ax_w[mask] = ax_w_list[i][mask]
            ax_v[mask] = ax_v_list[i][mask]
            ax_u[mask] = ax_u_list[i][mask]

        z = 1.0 + r2 * (cosθmax - 1.0)
        x = torch.cos(phi) * torch.sqrt(1.0 - z ** 2)
        y = torch.sin(phi) * torch.sqrt(1.0 - z ** 2)

        ray_dir = ax_u * x.reshape(-1, 1) + ax_v * y.reshape(-1, 1) + ax_w * z.reshape(-1, 1)
        return ray_dir


class mixed_pdf(PDF):
    """Probability density Function"""

    def __init__(self, shape, pdf1, pdf2, pdf1_weight=0.5):

        self.pdf1_weight = torch.as_tensor(pdf1_weight)
        self.pdf2_weight = 1.0 - self.pdf1_weight
        self.shape = shape
        self.pdf1 = pdf1
        self.pdf2 = pdf2

    def value(self, ray_dir):
        val = (
            self.pdf1.value(ray_dir) * self.pdf1_weight
            + self.pdf2.value(ray_dir) * self.pdf2_weight
        )
        assert max(val) <= 1
        assert min(val) >= 0
        return val

    def generate(self):
        mask = distr.Uniform(torch.zeros(self.shape), 1.).sample()
        return torch.where(
            (mask < self.pdf1_weight).reshape(-1, 1), self.pdf1.generate(), self.pdf2.generate()
        )


class normal_pdf(PDF):
    """Normal distribution"""

    def __init__(self, mu, sigma=1):
        self.mu = mu  # mean/centre
        self.sigma = sigma  # standard deviation

#     def value(self, x: vec3) -> np.array:
#         """Evaluate the normal at the given point."""
#         epsilon = 1e-5
#         val_x = norm.pdf(x.x, self.mu.x, self.sigma) * epsilon
#         val_y = norm.pdf(x.y, self.mu.y, self.sigma) * epsilon
#         val_z = norm.pdf(x.z, self.mu.z, self.sigma) * epsilon
#         val = val_x * val_y * val_z
#         assert max(val) <= 1
#         assert min(val) >= 0
#         return val
# 
#     def log_value(self, x: vec3) -> np.array:
#         """Evaluate the log normal at the given point."""
#         epsilon = 1e-5
#         val_x = norm.logpdf(x.x, self.mu.x, self.sigma) + np.log(epsilon)
#         val_y = norm.logpdf(x.y, self.mu.y, self.sigma) + np.log(epsilon)
#         val_z = norm.logpdf(x.z, self.mu.z, self.sigma) + np.log(epsilon)
#         val = val_x + val_y + val_z
#         assert max(val) <= 0
#         return val

    def generate(self):
        return distr.Normal(self.mu, self.sigma).sample()
# 
# 
# def random_in_unit_spherical_caps(shape, origin, importance_sampled_list):
# 
#     l = len(importance_sampled_list)
# 
#     mask = (np.random.rand(shape) * l).astype(int)
#     mask_list = [None] * l
# 
#     cosθmax_list = [None] * l
#     ax_u_list = [None] * l
#     ax_v_list = [None] * l
#     ax_w_list = [None] * l
# 
#     for i in range(l):
# 
#         ax_w_list[i] = (importance_sampled_list[i].center - origin).normalize()
#         a = vec3.where(np.abs(ax_w_list[i].x) > 0.9, vec3(0, 1, 0), vec3(1, 0, 0))
#         ax_v_list[i] = ax_w_list[i].cross(a).normalize()
#         ax_u_list[i] = ax_w_list[i].cross(ax_v_list[i])
#         mask_list[i] = mask == i
# 
#         target_distance = np.sqrt(
#             (importance_sampled_list[i].center - origin).dot(
#                 importance_sampled_list[i].center - origin
#             )
#         )
# 
#         cosθmax_list[i] = np.sqrt(
#             1
#             - np.clip(
#                 importance_sampled_list[i].bounded_sphere_radius / target_distance,
#                 0.0,
#                 1.0,
#             )
#             ** 2
#         )
# 
#     phi = np.random.rand(shape) * 2 * np.pi
#     r2 = np.random.rand(shape)
# 
#     cosθmax = np.select(mask_list, cosθmax_list)
#     ax_w = vec3.select(mask_list, ax_w_list)
#     ax_v = vec3.select(mask_list, ax_v_list)
#     ax_u = vec3.select(mask_list, ax_u_list)
# 
#     z = 1.0 + r2 * (cosθmax - 1.0)
#     x = np.cos(phi) * np.sqrt(1.0 - z ** 2)
#     y = np.sin(phi) * np.sqrt(1.0 - z ** 2)
# 
#     ray_dir = ax_u * x + ax_v * y + ax_w * z
# 
#     PDF = 0.0
#     for i in range(l):
#         PDF += np.where(
#             ray_dir.dot(ax_w_list[i]) > cosθmax_list[i],
#             1 / ((1 - cosθmax_list[i]) * 2 * np.pi),
#             0.0,
#         )
#     PDF = PDF / l
# 
#     return ray_dir, PDF
# 
# 
# def random_in_unit_spherical_cap(shape, cosθmax, normal):
# 
#     ax_w = normal
#     a = vec3.where(np.abs(ax_w.x) > 0.9, vec3(0, 1, 0), vec3(1, 0, 0))
#     ax_v = ax_w.cross(a).normalize()
#     ax_u = ax_w.cross(ax_v)
# 
#     phi = np.random.rand(shape) * 2 * np.pi
#     r2 = np.random.rand(shape)
# 
#     z = 1.0 + r2 * (cosθmax - 1.0)
#     x = np.cos(phi) * np.sqrt(1.0 - z ** 2)
#     y = np.sin(phi) * np.sqrt(1.0 - z ** 2)
# 
#     return ax_u * x + ax_v * y + ax_w * z
