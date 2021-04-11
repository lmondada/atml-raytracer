from . import Material
from ..textures import solid_color


class Emissive(Material):
    def __init__(self, color, **kwargs):

        assert color.shape == (3,)
        self.texture_color = solid_color(color)

        super().__init__(**kwargs)

    def get_color(self, scene, ray, hit, max_index):
        n_rays = ray.color.shape[0]
        diff_color = self.texture_color.get_color(hit).repeat(n_rays, 1)
        ray.color = diff_color
        return ray
