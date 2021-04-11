#
# Generalized Galton board example.
#
import sys
import numpy as np
import torch
from torch import nn
import pyro
from .pyro_simulator import PyroSimulator

import sys
sys.path.insert(1, '../../pyro_simulator')
from raytracer import *

class RaytracerSimulator(PyroSimulator):

    """
    Generalized Galton board example from arXiv:1805.12244.


    Has one parameter:
    theta

    Three hyperparameters:
    n_row = number of rows
    n_nails = number of nails
    start_pos = starting position (default = n_nails / 2)
    """

    def __init__(self, samples_per_pixel=8):
        super(RaytracerSimulator, self).__init__()

        self.samples_per_pixel = samples_per_pixel

    def _set_scene(self, purity):
        self.scene = Scene(ambient_color=torch.zeros(3))
        self.scene.add_Camera(
            screen_width=128,
            screen_height=128,
            look_from=torch.tensor([40., 400., 300.]),
            look_at=torch.tensor([500., 0., -500.]),
            focal_distance=1.0,
            field_of_view=30,
        )

        # define materials to use
        gray_diffuse = Diffuse(diff_color=torch.tensor([.43, .43, .43]))
        white_diffuse = Diffuse(diff_color=torch.tensor([.73, .73, .73]))
        emissive_white = Emissive(color=torch.tensor([.9, .9, .9]))
        glass = Refractive(n=torch.tensor([1.5 + 0j, 1.5 + 0j, 1.5 + 0j]), purity=purity)

        # this is the light
        self.scene.add(
            Plane(
                material=emissive_white,
                center=torch.tensor([213 + 130 / 2, 400, -227.0 - 105 / 2]),
                width=130.0,
                height=105.0,
                u_axis=torch.tensor([1.0, 0.0, 0]),
                v_axis=torch.tensor([0.0, 0, 1.0]),
            ),
            importance_sampled=True,
        )
        self.scene.add(
            Plane(
                material=emissive_white,
                center=torch.tensor([5, 225, -227.0]),
                width=130.0,
                height=105.0,
                u_axis=torch.tensor([0.0, 1.0, 0]),
                v_axis=torch.tensor([0.0, 0, 1.0]),
            ),
            importance_sampled=True,
        )

        self.scene.add(
            Plane(
                material=white_diffuse,
                center=torch.tensor([555 / 2, 555 / 2, -555.0]),
                width=555.0,
                height=555.0,
                u_axis=torch.tensor([0.0, 1.0, 0]),
                v_axis=torch.tensor([1.0, 0, 0.0]),
            )
        )

        self.scene.add(
            Plane(
                material=gray_diffuse,
                center=torch.tensor([0, 555 / 2, -555 / 2]),
                width=555.0,
                height=555.0,
                u_axis=torch.tensor([0.0, 1.0, 0]),
                v_axis=torch.tensor([0.0, 0, -1.0]),
            )
        )

        self.scene.add(
            Plane(
                material=gray_diffuse,
                center=torch.tensor([555.0, 555 / 2, -555 / 2]),
                width=555.0,
                height=555.0,
                u_axis=torch.tensor([0.0, 1.0, 0]),
                v_axis=torch.tensor([0.0, 0, -1.0]),
            )
        )

        self.scene.add(
            Plane(
                material=white_diffuse,
                center=torch.tensor([555 / 2, 555, -555 / 2]),
                width=555.0,
                height=555.0,
                u_axis=torch.tensor([1.0, 0.0, 0]),
                v_axis=torch.tensor([0.0, 0, -1.0]),
            )
        )

        self.scene.add(
            Plane(
                material=white_diffuse,
                center=torch.tensor([555 / 2, 0.0, -555 / 2]),
                width=555.0,
                height=555.0,
                u_axis=torch.tensor([1.0, 0.0, 0]),
                v_axis=torch.tensor([0.0, 0, -1.0]),
            )
        )

        self.scene.add(
            Sphere(
                material=glass,
                center=torch.tensor([320.5, 100, -240]),
                radius=100,
                shadow=False,
                max_ray_depth=2,
                # mc=True,
            ),
            importance_sampled=True,
        )

    def forward(self, inputs):
        purity = inputs

        self._set_scene(purity)
        all_colors, all_pixel_indices, all_log_p_offsets = self.scene.render(
            samples_per_pixel=self.samples_per_pixel,
            progress_bar=True
        )

        outputs = torch.cat((all_colors, all_pixel_indices.reshape(-1, 1), all_log_p_offsets.reshape(-1, 1)), dim=1)
        return outputs

        # Define a pyro distribution
        # distribution_u = pyro.distributions.Uniform(0, 1).expand([num_samples])
        # dist_bern = pyro.distributions.Bernoulli(probs=a)

        # Run and mine gold
        # left/right decisions are based on value of theta
        # log_pxz based on value of theta


        # begin = torch.zeros(num_samples).fill_(self.start_pos)
        pos = torch.zeros(num_samples).fill_(self.start_pos)

        # begin = np.empty(num_samples)
        # pos = np.empty(num_samples)

        # begin.fill(self.n_nails // 2)
        # pos.fill(self.n_nails // 2)

        # z = []
        z=torch.tensor(0,dtype=torch.float)

        while z < self.n_rows:
            '''
            Parametrization of the Galton Board
            
            level 0 =   0   1   2   3   ...  29  30  ...
            level 1 = 0   1   2   3   ...  29  30  ...
            level 2 =   0   1   2   3   ...  29  30  ...
            '''

            # print('pos before calling threshold = ',pos)
            # print('begin before calling threshold = ', begin)
            level = z
            print('Level = ',level)

            tmp_prob = self.nail_positions(theta, level=level, nail=pos)

            if level % 2 == 0:  # For the 1st nail (pos=0) the prob. of going left is 0 and for the last one (pos=n_nail) it is 1)
                tmp_prob[pos == float(self.n_nails)] = 1.0
            else:
                tmp_prob[pos == 0.] = 0.0

            t =tmp_prob

            # t = self.threshold(theta, (begin, z))
            # t is the probability of going left. At first z is empty so we don't change pos, and use the begin value.



            # Left indices
            dist_bern = pyro.distributions.Bernoulli(probs=t)

            # print('Sample =',dist_bern.sample())
            draw = pyro.sample("u" + str(z), dist_bern)  # We draw a number for each ball from the distribution_u


            print('t = ', t)
            print('pyro sample = ',draw) # We sample from the pyro distribution
            print('---')


            # # going left
            print('pos before = ', pos)

            # draw==1 means going left and draw==0 going right
            if level % 2 == 0:  # even rows
                pos[draw==0] = pos[draw==0] + 1 # We move to the right when ind_list[i]==0 (When the sampled value is greater than the prob. of going left).
            else:  # odd rows
                pos[draw==1] = pos[draw==1] - 1 # We move to the left when ind_list[i]==1


            print('pos after = ', pos)
            print('---' * 40)
            print('---' * 40)

            z+=1

        x = pos

        return x
"""
    def suggested_parameters(self):

        return np.array([0.1, 50])

    def suggested_times(self):

        return np.linspace(0, 100, 100)
        
"""
