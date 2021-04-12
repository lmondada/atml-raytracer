## First dataset

This folder contains the code to generate the original dataset produced.
Note that this version of the dataset did not compute any joint scores.
Check the later dataset in the folder `../second_dataset` for a similar, albeit
smaller dataset that includes joint scores.

### Dataset
#### Preview
<img src="/docs/images/first_dataset.png">

#### Downloading and reproducing the dataset
The dataset is available for download [here](https://github.com/lmondada/atml-raytracer/releases/download/first_dataset/data.zip).
To reproduce the dataset, use this repo at the following
[tagged commit](https://github.com/lmondada/atml-raytracer/tree/first_dataset) (Release tag `first_dataset`).

### Data generation
The `generate_data.py` script will run the raytracer for a simple scene with a
transparent refractive sphere in its middle, for different purities of the sphere material
(with purities in `[0.5, 0.55, 0.6, 0.65, ..., 0.95]`).
The output are 128x128 images, saved as `data/purity_0.XX/result.png`.
The data will be saved in CSV files in subdirectories `data/purity_0.XX/all_rays.csv`.
On a 1.4 GHz Quad-Core Intel Core i5 processor, the data generation for each render takes
about 30min.

All the data can be gathered into a single CSV file `data/all_rays.csv` by running the
`combine.py` script after the data generation is complete.
An image combining all images can also be produced using `combine_images.py`.

### Data description
Each row in the CSV files correspond to a single ray simulation with the following features:
 - `pixel_index,color,log_p_z,log_p_z_ref
 - `pixel_index`: the pixel the ray hit, as an integer between `0` and `16384` (`128^2`).
 - `color`: the color of the ray hitting the camera, as a grayscale between `0` and `1`.
 - `log_p_z`: the log probability `log P(x, z | θ)` of that particular ray, where `x` is the
    color of the ray, `z` is the latent state (given by the position and color of the ray at each bouncing point)
    and `θ` is the purity of the sphere material.
 - `log_p_z_ref` the log probability `log P(x, z | θ)` at the reference purity `θ_ref = 0.75`.
