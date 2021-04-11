import torch
import pandas as pd
from PIL import Image
import progressbar
import time

from simulators import RaytracerSimulator
from raytracer.utils import colour_functions as cf

# theta = torch.tensor(5000*[[3.]])
# theta_ref = torch.tensor(5000*[[0.7]])

purity_ref = torch.tensor(0.75)
height = width = 12

samples_per_pixel = 5

def get_rays(purity):
    print("Rendering...")
    all_colors = torch.tensor([]).reshape(0, 3)
    all_pixel_indices = []
    all_joint_scores = []
    all_joint_log_ratios = []

    simulator = RaytracerSimulator(height=height, width=width)
    n_pixels = len(simulator.all_rays)

    bar = progressbar.ProgressBar(maxval=samples_per_pixel * n_pixels)


    bar.start()
    t0 = time.time()
    try:
        for j in range(samples_per_pixel):
            for i in  range(n_pixels):
                simulator.set_pixel(i)
                color, joint_score, joint_log_ratio = simulator.augmented_data(purity, purity, purity_ref)
                # save the data
                all_colors = torch.cat((all_colors, color))
                all_pixel_indices.append(i)
                all_joint_scores.append(joint_score)
                all_joint_log_ratios.append(joint_log_ratio)
                # all_log_p_offsets = torch.cat((all_log_p_offsets, rays.log_p_offset))
                bar.update(n_pixels * j + i)
    finally:
        bar.finish()
        t1 = time.time()
        print(f'Took {t1-t0}s')
        print('---'*5)

        data = pd.DataFrame(data={
            'color': all_colors,
            'pixel_index': all_pixel_indices,
            'joint_score': all_joint_scores,
            'joint_log_ratio': all_joint_log_ratios
        })
        data.to_csv('saved.csv')


        # Aggregate the colours per pixel
        color_RGBlinear = torch.stack([
            all_colors[torch.tensor(all_pixel_indices) == i].mean(dim=0)
            for i in torch.arange(min(all_pixel_indices), max(all_pixel_indices) + 1)
        ])
        # gamma correction
        color_corrected = cf.sRGB_linear_to_sRGB(color_RGBlinear)

        img_RGB = (255 * torch.clip(color_corrected, 0., 1.)).reshape(
                width, height, 3
        ).type(torch.uint8)
        im = Image.merge("RGB", [
            Image.fromarray(img_RGB[..., i].numpy(), "L") for i in range(3)
        ])
        im.save("result.png")
        im.show()

if __name__ == '__main__':
    get_rays(torch.tensor(0.8))
## x, joint_score, joint_log_ratio = simulator.augmented_data(theta,None, None)



#
# print(torch.mean(joint_score, dim=0))
# print(torch.mean(torch.exp(joint_log_ratio), dim=0))

#printer = pprint.PrettyPrinter(indent=4)
#printer.pprint(simulator.trace(inputs).nodes)

# plt.hist(output.numpy(), 31,range=[0,31])
# plt.xlabel(r"$x$")
# plt.ylabel("Frequency")
#
# plt.show()
