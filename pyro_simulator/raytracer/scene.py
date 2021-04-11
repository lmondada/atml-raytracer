from PIL import Image
import numpy as np
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
        # rays_per_batch = len(self.camera.get_ray(self.n))
        # batch_size = batch_size or np.ceil(samples_per_pixel / n_proc).astype(int)
        #
        # all_rays_batched = batch_rays(all_rays, batch_size)
        args = [(ray, copy.deepcopy(self)) for ray in all_rays]

        def compute_cols(ray):
            # Aggregate the colours per pixel
            av_col_per_pixel = torch.stack([
                ray.color[ray.pixel_index == i].mean(dim=0) # this should take log_p_offset into account
                for i in range(min(ray.pixel_index), max(ray.pixel_index) + 1)
            ])
            sum_col_per_pixel = torch.stack([
                ray.color[ray.pixel_index == i].sum(dim=0)
                for i in range(min(ray.pixel_index), max(ray.pixel_index) + 1)
            ])
            # combined_cols = sum_col_per_pixel[0].append(tuple(sum_col_per_pixel[1:]))
            # # rescale colors
            # max_val = max(combined_cols.max().to_array())
            # print("max col value is ", max_val)
            # if max_val > 1:
            #     combined_cols = combined_cols / max_val
            # combined_cols.clip(0.0, 1.0)

            # Aggregate the mining statistics
#             mining["col_probs"] = np.concatenate((mining["col_probs"], ray.log_p_z))
#             mining["col_probs_ref"] = np.concatenate(
#                 (mining["col_probs_ref"], ray.log_p_z_ref)
#             )

            return av_col_per_pixel

#         def refine(dust):
#             """Sort the gold out from the dust:
#             Since we collected the data backwards, we need to reverse it to compute the actual statistics.
#             Variables:
#              - Z: ray state for each ray in the image that we are tracking
#              - Z_i: ray states after i bounces (counting from the light source)
#              - z, z_i: state for a single ray (after i bounces)
#              - x: final image
# 
#             We want to collect:
#              - Joint Score: t(x,Z|θ) = log p(x,Z|θ) = SUM[ log p(Z_i|θ,Z_{<=i}) ] + log p(x|θ,Z), for each θ_i.
#              - Joint Likelihood Ratio: r(x,Z|θ_0, θ_1) = p(x,Z|θ_0)/p(x,Z|θ_1)
# 
#             We have accumulated:
#              - p(z_i <- z_i-1|θ) transition probability for each i for each ray.
# 
#             We know:
#              - p(x|θ,Z) = 1 (since x is just an average of the final ray colours/positions that land in the detector).
#              - > p(x,Z|θ) = p(Z|θ) by law of conditional probability.
#              - > r(x,Z|θ_0, θ_1) = p(Z|θ_0)/p(Z|θ_1) = exp[js(θ_0) - js(θ_1)]
# 
#             for each ray:
#              - p(z_i|z<i,θ) = p(z_i <- z_i-1|θ) ie prob that we transition to z_i from z_i-1
#              - > log p(z|θ) = SUM_i log p(z_i -> z_i+1|θ) transition/bounce probability
# 
#             combining the rays:
#              - p(Z_i|θ,z_<i) = PROD_z p(z_i|θ,z<i) since each ray is independent of the others.
#              - > log p(Z_i|θ,Z_<i) = SUM_z log p(z_i|θ,z<i) = SUM_z SUM_j<i log p(z_j <- z_j-1|θ)
# 
#             So mining the gold is just a case of summing over the log probs of complete paths.
#             We can ignore incomplete paths (paths that didn't make it to the detector in time).
#             This is ok to do, since they contribute nothing to the final image anyway (they add 0).
#             """
#             # first, we need to filter out runs that didn't hit the light (log prob 1)
#             log_clean_pz_ref = dust["col_probs_ref"][dust["col_probs"] != 1.0]
#             log_clean_pz = dust["col_probs"][dust["col_probs"] != 1.0]
# 
#             assert all(log_clean_pz <= 1e-7)
#             assert all(log_clean_pz_ref <= 0)
# 
#             js_0 = sum(log_clean_pz)
#             js_1 = sum(log_clean_pz_ref)
#             jlr = np.exp(js_0 - js_1)
# 
#             print("Joint score:", js_0)
#             print("Reference joint score:", js_1)
#             print("Joint likelihood ratio:", jlr)
# 
#             return js_0, js_1, jlr

        # Keep track of all the gold dust we are mining
        mined_dust = {
            "col_probs": [],
            "col_probs_ref": [],
        }

        bar = progressbar.ProgressBar(maxval=len(args))
        all_colors = torch.tensor([]).reshape(0, 3)
        all_pixel_indices = torch.tensor([])
        all_log_p_offsets = torch.tensor([])

        with Pool(processes=n_proc) as pool:
            bar.start()
            for i, (rays, _) in enumerate(
                pool.imap_unordered(get_raycolor_tuple, args)
            ):
#                     color = compute_cols(rays)
#                     color_RGBlinear += color
                # save the data
                all_colors = torch.cat((all_colors, rays.color))
                all_pixel_indices = torch.cat((all_pixel_indices, rays.pixel_index))
                all_log_p_offsets = torch.cat((all_log_p_offsets, rays.log_p_offset))
                bar.update(i)
        bar.finish()
#             if save_csv is not None:
#                 # backup all data in file
#                 with open(save_csv, "w", newline="") as csvfile:
#                     fieldnames = ["pixel_index", "color", "log_p_z", "log_p_z_ref"]
#                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                     writer.writeheader()
#                     for ray_data in all_rays_data:
#                         if ray_data["log_p_z"] == 1.0:
#                             continue
#                         writer.writerow(ray_data)
        return all_colors, all_pixel_indices, all_log_p_offsets

        # average samples per pixel (antialiasing)
        color_RGBlinear = color_RGBlinear / samples_per_pixel
        # gamma correction
        color = cf.sRGB_linear_to_sRGB(color_RGBlinear)

        print("Render Took", time.time() - t0)

        img_RGB = (255 * torch.clip(color, 0., 1.)).reshape(
                self.camera.screen_height, self.camera.screen_width, 3
        ).type(torch.uint8)
        return Image.merge("RGB", [
            Image.fromarray(img_RGB[..., i].numpy(), "L") for i in range(3)
        ]), {}
#         img_RGB = []
#         for c in color:
#             # average ray colors that fall in the same pixel. (antialiasing)
#             img_RGB += [
#                 Image.fromarray(
#                     (
#                         255
#                         * np.clip(c, 0, 1).reshape(
#                             (self.camera.screen_height, self.camera.screen_width)
#                         )
#                     ).astype(np.uint8),
#                     "L",
#                 )
#             ]
# 
#         gold_bars = refine(mined_dust)
#         return Image.merge("RGB", img_RGB), gold_bars

#     def get_distances(
#         self,
#     ):  # Used for debugging ray-primitive collisions. Return a grey map of objects distances.
# 
#         print("Rendering...")
#         t0 = time.time()
#         color_RGBlinear = get_distances(self.camera.get_ray(self.n), scene=self)
#         # gamma correction
#         color = color_RGBlinear.to_array()
# 
#         print("Render Took", time.time() - t0)
# 
#         img_RGB = [
#             Image.fromarray(
#                 (
#                     255
#                     * np.clip(c, 0, 1).reshape(
#                         (self.camera.screen_height, self.camera.screen_width)
#                     )
#                 ).astype(np.uint8),
#                 "L",
#             )
#             for c in color
#         ]
#         return Image.merge("RGB", img_RGB)
