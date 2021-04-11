from math import pi
import torch

from ..utils.random import spherical_caps_pdf, cosine_pdf, mixed_pdf
from ..ray import Ray, get_raycolor
from . import Material
from ..textures import solid_color

class Diffuse(Material):
    def __init__(
        self,
        diff_color,
        diffuse_rays=10,
        ambient_weight=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert diff_color.shape == (3,)
        self.diff_texture = solid_color(diff_color)

#         elif isinstance(diff_color, texture):
#             self.diff_texture = diff_color  # Colour of the material

#         if isinstance(diff_color_ref, vec3):
#             self.diff_texture_ref = solid_color(diff_color_ref)
#         elif isinstance(diff_color_ref, texture):
#             self.diff_texture_ref = diff_color_ref  # Colour of reference material

        self.diffuse_rays = diffuse_rays  # Number of rays to average over
        # Number of bounces on diffuse surfaces to compute
        self.max_diffuse_reflections = 2
        # How much of the ambient colour leaks into this object
        self.ambient_weight = ambient_weight  
#         self.ambient_weight_ref = ambient_weight_ref

    def get_pdf(self, size, N_repeated, nudged_repeated, scene):
        """Get the probability density function for diffuse reflections for the current hit."""
        pdf1 = cosine_pdf(size, N_repeated)
        pdf2 = spherical_caps_pdf(size, nudged_repeated, scene.importance_sampled_list)

        if not scene.importance_sampled_list:
            s_pdf = cosine_pdf(size, N_repeated)
        else:
            ambient_weight = self.ambient_weight
            s_pdf = mixed_pdf(size, pdf1, pdf2, ambient_weight)

        return s_pdf

    def get_color(self, scene, ray, hit, max_index):
        """Get the colour of this material for the given ray and intersect details."""

        hit.point = ray.origin + ray.dir * hit.distance.reshape(-1, 1)  # intersection point
        N = hit.material.get_Normal(hit)  # normal

        diff_color = self.diff_texture.get_color(hit)  # Own colour at this point
        # diff_color_ref = self.diff_texture_ref.get_color(hit)

        # Default in case the ray doesn't reach a light source.
        color = torch.zeros(3)

        # Count the number of incoming rays for indexing purposes.
        n_rays_in = ray.color.shape[0]

        if ray.diffuse_reflections < 1:
            """First diffuse reflection, average multiple rays."""
            n_diffuse_rays = self.diffuse_rays
        elif ray.diffuse_reflections < self.max_diffuse_reflections:
            """when ray.diffuse_reflections > 1 we just call one diffuse ray to solve rendering equation 
            (otherwise is too slow)"""
            n_diffuse_rays = 1

        else:
            """Max diffuse reflections reached; don't compute the ray any further.
            Set probabilities < 0 to discount from later calculation."""
#             ray.log_p_z = np.full(ray.log_p_z.shape, 1)
#             ray.log_p_z_ref = np.full(ray.log_p_z_ref.shape, 1)
            ray.color = color.repeat(n_rays_in, 1)
            return ray

        # Compute n_diffuse_rays ray colours:
        nudged = hit.point + N * 0.000001
        N_repeated = N.repeat(n_diffuse_rays, 1)

        if ray.n.shape == 1:
            n_repeated = ray.n.repeat(ray.log_p_z.shape[0] * n_diffuse_rays, 1)
        else:
            n_repeated = ray.n.repeat(n_diffuse_rays, 1)

        nudged_repeated = nudged.repeat(n_diffuse_rays, 1)
        hit_repeated = hit.point.repeat(n_diffuse_rays, 1)
#         log_pz_repeated = np.repeat(ray.log_p_z, n_diffuse_rays)
#         log_pz_ref_repeated = np.repeat(ray.log_p_z_ref, n_diffuse_rays)

        size = N.shape[0] * n_diffuse_rays

        # Probability densities we will sample from:
        s_pdf = self.get_pdf(size, N_repeated, nudged_repeated, scene)
        # s_pdf_ref = self.get_pdf(size, N_repeated, nudged_repeated, scene, is_ref=True)

        # Sample n_diffuse_rays directions where the current ray could have come from:
        ray_dir = s_pdf.generate()
        # ray_dir_ref = s_pdf_ref.generate()

        # Mine the probabilities of having sampled the directions we sampled from this material:
        PDF_val = s_pdf.value(ray_dir)
        # PDF_val_ref = s_pdf_ref.value(ray_dir)

        # Generate indices for the new rays
        new_ray_indices = torch.tensor(
            [
                # Keep the old indices in the right places
                ray.ray_index[i] if j == 0 else max_index + i * (n_diffuse_rays - 1) + j
                for i in range(n_rays_in)
                for j in range(n_diffuse_rays)
            ]
        )
        new_max_index = (
            max_index + n_rays_in * (n_diffuse_rays - 1)
            if n_diffuse_rays > 1
            else max_index
        )

        new_ray_deps = ray.ray_index.repeat(n_diffuse_rays).reshape(-1, 1)

#         assert np.all(log_pz_ref_repeated != 1.0)
#         assert np.all(log_pz_repeated != 1)
#         assert np.all((log_pz_repeated + np.log(PDF_val) < 1e-7))
#         assert np.all(
#             (log_pz_ref_repeated + np.log(PDF_val_ref) < 1e-7)
#             | (log_pz_ref_repeated == 1.0)
#         )

        # Recurse to compute the colour of the sampled rays
        out_ray, _ = get_raycolor(
            Ray(
                pixel_index=ray.pixel_index.repeat(n_diffuse_rays),
                ray_index=new_ray_indices,
                ray_dependencies=torch.cat(
                    (
                        ray.ray_dependencies.repeat(n_diffuse_rays, 1),
                        new_ray_deps,
                    ), dim=-1,
                ),
                origin=nudged_repeated,
                dir=ray_dir,
                depth=ray.depth + 1,
                n=n_repeated,
                log_p_offset=ray.log_p_offset.repeat(n_diffuse_rays),
#                 log_trans_probs=log_pz_repeated + np.log(PDF_val),
#                 log_trans_probs_ref=log_pz_ref_repeated + np.log(PDF_val_ref),
                color=ray.color.repeat(n_diffuse_rays, 1),
                reflections=ray.reflections + 1,
                transmissions=ray.transmissions,
                diffuse_reflections=ray.diffuse_reflections + 1,
            ),
            scene,
            new_max_index,
        )

        color_temp = out_ray.color
        # dot product of each ray with the normal
        NdotL = torch.clip(torch.sum(ray_dir * N_repeated, dim=1), 0.0, 1.0)

        # update values to account for any new rays.
        indexing_order = [
            torch.where(ray.ray_index == pos)[0][0]
            for pos in out_ray.ray_dependencies[:, -1]
        ]

        PDF_val = torch.tensor([PDF_val[pos] for pos in indexing_order])
        # PDF_val_ref = np.array([PDF_val_ref[round(pos)] for pos in indexing_order])
        NdotL = torch.tensor([NdotL[pos] for pos in indexing_order])

        # We have consumed this dependency layer so can now remove it.
        out_ray.ray_dependencies = out_ray.ray_dependencies[:, :-1]

        # Weight this colour by the probability of having sampled each ray
        # We use the Lambertian BRDF
        color_temp = color_temp * (NdotL / PDF_val).reshape(-1, 1) / pi
        # I am not sure this change is justified, but its kinda necessary
        # color_temp_ref = color_temp * NdotL / PDF_val_ref / np.pi

        # Collect the ray colors
        n_rays = out_ray.color.shape[0]
        out_ray.color = diff_color * color_temp
        # color_ref = color.repeat(n_rays) + diff_color_ref * color_temp_ref

        # Transition probabilities include the colours
        # since this is computed deterministically from the ray directions and theta,
        # this will simply make the ref prob 0 if color_temp and color_temp_ref do not match for a given ray.

        return out_ray
