## Second dataset

This folder contains the code to generate the final dataset produced
and used in the benchmarks in the report.

### Data generation
The `generate_data.py` script will run the raytracer for a simple scene with a
transparent refractive sphere in its middle, for different
refractive indices `n` of the sphere material
(`n = 1.5 + 1e-X`, with `X` in `[-7, -6, ..., 0]`)
and a material purity `0.9`.
The output are 50x50 images, saved as `data/purity_0.9_n_XX/result.png`.
The data will be saved in CSV files in subdirectories `data/purity_0.9_n_XX/all_rays.csv`.

All the data can be gathered into a single CSV file `data/all_rays.csv` by running the
`combine.py` script after the data generation is complete.

### Data description
Each row in the CSV files correspond to a single ray simulation with the following features:
 - `ray pixel index`: the pixel the ray hit, as an integer between `0` and `2500` (`50^2`).
 - `ray color`: the color of the ray hitting the camera, as a grayscale between `0` and `1`.
 - `joint likelihood ratio`: the joint likelihood ratio `P(x, z | θ) / P(x, z | θ_ref)` of that particular ray, where `x` is the
    color of the ray, `z` is the latent state (given by the position and color of the ray at each bouncing point),
    `θ` is the refractive index of the sphere material and `θ_ref = 1.4 + 1e-4j` is the reference refractive index.
 - `joint score`: the joint score `∇_θ log P(x, z | θ)` evaluated
at the purity θ.
 - `reference joint score`: the joint score `∇_θ log P(x, z | θ)` evaluated
at the reference refractive index `θ = 1.5 + 1e-4j`.
