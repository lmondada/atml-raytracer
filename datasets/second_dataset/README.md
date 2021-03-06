## Second dataset

This folder contains the code to generate the final dataset produced
and used in the benchmarks in the report.

### Downloading and reproducing the dataset
The dataset is available for download [here](https://github.com/lmondada/atml-raytracer/releases/download/v2/all_rays.csv).
To reproduce the dataset, use this repo at the following
[[tagged commit](https://github.com/lmondada/atml-raytracer/tree/v2)] (Release tag `second_dataset`).


### Data generation
The `generate_data.py` script will run the raytracer for a simple scene with a
transparent refractive sphere in its middle, for different
refractive indices `n` of the sphere material
(`n = X + 0j`, with `X` in `[0.1, 0.3, ..., 0.9, 1, 1.2,  ..., 2.4]`)
and a material purity `0.8`.
The output are 50x50 images, saved as `data/purity_0.8_n_XX/result.png`.
The data will be saved in CSV files in subdirectories `data/purity_0.8_n_XX/all_rays.csv`.

All the data can be gathered into a single CSV file `data/all_rays.csv` by running the
`combine.py` script after the data generation is complete.

### Data description
###### Note that the dataset used in the report is labelled 'data_n_real', varying only the real component of n.
###### The main dataset varies purity and both components of n relative to a fixed reference value.

Each row in the CSV files correspond to a single ray simulation with the following features:
 - `ray pixel index`: the pixel the ray hit, as an integer between `0` and `2500` (`50^2`).
 - `ray color`: the color of the ray hitting the camera, as a grayscale between `0` and `1`.
 - `joint likelihood ratio`: the joint likelihood ratio `P(x, z | θ) / P(x, z | θ_ref)` of that particular ray, where `x` is the
    color of the ray, `z` is the latent state (given by the position and color of the ray at each bouncing point),
    `θ` is the refractive index of the sphere material and `θ_ref = 1.5 + 0j` is the reference refractive index.
 - `joint score`: the joint score `∇_θ log P(x, z | θ)` evaluated
at the purity θ.
 - `reference joint score`: the joint score `∇_θ log P(x, z | θ)` evaluated
at the reference refractive index `θ = 1.5 + 0j`.
