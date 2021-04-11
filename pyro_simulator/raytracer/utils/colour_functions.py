import torch

def sRGB_linear_to_sRGB(rgb_linear):

    """sRGB standard for gamma inverse correction."""
    rgb = torch.where(
        rgb_linear <= 0.00304,
        12.92 * rgb_linear,
        1.055 * torch.pow(rgb_linear, 1.0 / 2.4) - 0.055,
    )

    # clip intensity if needed (rgb values > 1.0) by scaling
    rgb_max = (
            rgb.max(dim=1)[0] + 0.00001  # avoid division by zero
    ).reshape(-1, 1)
    intensity_cutoff = 1.0
    rgb = torch.where(rgb_max > intensity_cutoff, rgb * intensity_cutoff / rgb_max, rgb)

    return rgb


def sRGB_to_sRGB_linear(rgb):

    """sRGB standard for gamma inverse correction."""
    rgb_linear = torch.where(
        rgb <= 0.03928, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4)
    )

    return rgb_linear
