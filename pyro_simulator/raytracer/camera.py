# from .utils.vector3 import vec3, rgb
import torch
import torch.nn.functional as F
from pyro import distributions as distr
from math import pi

from .utils.random import random_in_unit_disk
from .ray import Ray


class Camera:
    def __init__(
        self,
        look_from,
        look_at,
        screen_width=400,
        screen_height=300,
        field_of_view=90.0,
        aperture=0.0,
        focal_distance=1.0,
    ):
        self.screen_width = torch.as_tensor(screen_width)
        self.screen_height = torch.as_tensor(screen_height)
        self.aspect_ratio = self.screen_width / self.screen_height

        self.look_from = torch.as_tensor(look_from)
        self.look_at = torch.as_tensor(look_at)
        self.camera_width = torch.tan(torch.tensor(field_of_view) * pi / 180 / 2.0) * 2.0
        self.camera_height = self.camera_width / self.aspect_ratio

        # camera reference basis in world coordinates
        self.cameraFwd = F.normalize(look_at - look_from, dim=0)
        self.cameraRight = F.normalize(self.cameraFwd.cross(torch.tensor([0.0, 1.0, 0.0])), dim=0)
        self.cameraUp = self.cameraRight.cross(self.cameraFwd)

        # if you use a lens_radius >= 0.0 make sure that samples_per_pixel is a large number.
        # Otherwise you'll get a lot of noise
        self.lens_radius = torch.tensor(aperture) / 2.0
        self.focal_distance = torch.tensor(focal_distance)

        # Pixels coordinates in camera basis:
        self.x = torch.linspace(
            -self.camera_width / 2.0, self.camera_width / 2.0, self.screen_width
        )
        self.y = torch.linspace(
            self.camera_height / 2.0, -self.camera_height / 2.0, self.screen_height
        )

        # we are going to cast a total of screen_width * screen_height * samples_per_pixel rays
        # xx,yy store the origin of each ray in a 3d array where the first and second dimension
        # are the x,y coordinates of each pixel
        # and the third dimension is the sample index of each pixel
        xx, yy = torch.meshgrid(self.x, self.y)
        self.x = xx.flatten()
        self.y = yy.flatten()

    def get_ray(
        self, n
    ):  # n = index of refraction of scene main medium (for air n = 1.)

        # in each pixel, take a random position to avoid aliasing.
        width_ratio = self.camera_width / self.screen_width
        height_ratio = self.camera_height / self.screen_height
        x = self.x + distr.Uniform(
                low=-0.5 * width_ratio.repeat(len(self.x)),
                high=0.5 * width_ratio.repeat(len(self.x)),
        ).sample()
        y = self.y + distr.Uniform(
                low=-0.5 * height_ratio.repeat(len(self.y)),
                high=0.5 * height_ratio.repeat(len(self.y)),
        ).sample()

        # set ray direction in world space:
        rx, ry = random_in_unit_disk(x.shape[0])
        rx = rx.reshape(-1, 1)
        ry = ry.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        ray_origin = (
            self.look_from
            + self.cameraRight * rx * self.lens_radius
            + self.cameraUp * ry * self.lens_radius
        )
        ray_dir = F.normalize(
            self.look_from
            + self.cameraUp * y * self.focal_distance
            + self.cameraRight * x * self.focal_distance
            + self.cameraFwd * self.focal_distance
            - ray_origin,
            dim=-1
        )

        n_rays = x.shape[0]
        return Ray(
            pixel_index=torch.arange(n_rays),
            ray_index=torch.arange(n_rays),
            ray_dependencies=torch.arange(n_rays).reshape((n_rays, 1)),
            origin=ray_origin,
            dir=ray_dir,
            depth=0,
            n=torch.as_tensor(n),
            log_p_offset=torch.zeros(n_rays),
            # log_trans_probs=torch.zeros((n_rays)),
            # log_trans_probs_ref=np.zeros((n_rays)),
            color=torch.zeros(n_rays, 3),
            reflections=0,
            transmissions=0,
            diffuse_reflections=0,
        )
