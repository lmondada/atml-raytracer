from PIL import Image
import time
import copy
from multiprocessing import Pool, cpu_count
import csv
import torch

from .utils import colour_functions as cf
from .camera import Camera
from .utils.constants import *
from .ray import Ray, get_raycolor

import progressbar


def get_raycolor_tuple(x):
    return get_raycolor(*x)


def batch_rays(rays, batch_size):
    batches = []
    n_rays = len(rays)
    for ray_ind in range(0, n_rays, batch_size):
        batches.append(Ray.concatenate(rays[ray_ind : ray_ind + batch_size]))
    return batches


class Scene:
    def __init__(self, ambient_color=torch.tensor([0.01, 0.01, 0.01]), n=torch.tensor([1.0, 1.0, 1.0+0.j])):
        # n = index of refraction (by default index of refraction of air n = 1.)

        self.scene_primitives = []
        self.collider_list = []
        self.shadowed_collider_list = []
        self.Light_list = []
        self.importance_sampled_list = []
        self.ambient_color = ambient_color
        self.n = n
        self.importance_sampled_list = []

    def add_Camera(self, look_from, look_at, **kwargs):
        self.camera = Camera(look_from, look_at, **kwargs)

    def add_PointLight(self, pos, color):
        self.Light_list += [lights.PointLight(pos, color)]

    def add_DirectionalLight(self, Ldir, color):
        self.Light_list += [lights.DirectionalLight(Ldir.normalize(), color)]

    def add(self, primitive, importance_sampled=False):
        self.scene_primitives += [primitive]
        self.collider_list += primitive.collider_list

        if importance_sampled == True:
            self.importance_sampled_list += [primitive]

        if primitive.shadow == True:
            self.shadowed_collider_list += primitive.collider_list

#     def add_Background(self, img, light_intensity=0.0, blur=0.0, spherical=False):
# 
#         primitive = None
#         if spherical == False:
#             primitive = SkyBox(img, light_intensity=light_intensity, blur=blur)
#         else:
#             primitive = Panorama(img, light_intensity=light_intensity, blur=blur)
# 
#         self.scene_primitives += [primitive]
#         self.collider_list += primitive.collider_list

    def render(
        self, samples_per_pixel, progress_bar=False, batch_size=None, save_csv=None
    ):

        print("Rendering...")

        t0 = time.time()
        all_rays = [self.camera.get_ray(self.n) for i in range(samples_per_pixel)]
        color_RGBlinear = torch.zeros(all_rays[0].length, 3)

        n_proc = cpu_count()
        args = [(ray, copy.deepcopy(self)) for ray in all_rays]

        bar = progressbar.ProgressBar(maxval=len(args))
        all_colors = torch.tensor([]).reshape(0, 3)
        all_pixel_indices = torch.tensor([])
        all_log_p_offsets = torch.tensor([])

        with Pool(processes=n_proc) as pool:
            bar.start()
            for i, (rays, _) in enumerate(
                pool.imap_unordered(get_raycolor_tuple, args)
            ):
                # save the data
                all_colors = torch.cat((all_colors, rays.color))
                all_pixel_indices = torch.cat((all_pixel_indices, rays.pixel_index))
                all_log_p_offsets = torch.cat((all_log_p_offsets, rays.log_p_offset))
                bar.update(i)
        bar.finish()
        return all_colors, all_pixel_indices, all_log_p_offsets

    def render_single_ray(self, ray):

        # all_log_p_offsets = torch.tensor([])

        out_ray,_ = get_raycolor(ray, self)
        assert len(out_ray) == 1

        return out_ray
