import torch
import pandas as pd
from PIL import Image

from simulators import RaytracerSimulator
from raytracer.utils import colour_functions as cf

# theta = torch.tensor(5000*[[3.]])
# theta_ref = torch.tensor(5000*[[0.7]])

purity_ref = torch.tensor(0.75)


simulator = RaytracerSimulator(samples_per_pixel=3)

#simulator = Simulator(sensitivities=True)
# output = simulator(theta)


# print('Trace nodes =', simulator.trace(theta).nodes)
#
def get_rays(purity):
    color, indices, log_p_offset, joint_score, joint_log_ratio = simulator.augmented_data(purity, purity, purity_ref)

    print('color = ', color)
    print('indices = ', indices)
    print('log_p_offset = ', log_p_offset)
    print('joint_score = ',joint_score)
    print('joint_log_ratio= ',joint_log_ratio)
    print('---'*5)

    pd.DataFrame(data={
        'color': color,
        'pixel_index': indices,
        'joint_score': joint_score,
        'joint_log_ratio': joint_log_ratio
    })

# plot image

# Aggregate the colours per pixel
    color_RGBlinear = torch.stack([
        color[indices == i].mean(dim=0) # this should take log_p_offset into account
        for i in torch.arange(torch.min(indices), torch.max(indices) + 1)
    ])
# gamma correction
    color_corrected = cf.sRGB_linear_to_sRGB(color_RGBlinear)

    img_RGB = (255 * torch.clip(color_corrected, 0., 1.)).reshape(
            128, 128, 3
    ).type(torch.uint8)
    im = Image.merge("RGB", [
        Image.fromarray(img_RGB[..., i].numpy(), "L") for i in range(3)
    ])
    im.save("result.png")
    im.show()

if __name__ == '__main__':
    get_rays(torch.tensor(0.9))
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
