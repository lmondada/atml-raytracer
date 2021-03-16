# ATML Practical: Ray Tracer as a Galton Board
When trying to reproduce realistic physical properties of materials
in a ray tracer, the problem can be
phrased as a Monte Carlo sampling problem.
From that perspective, a ray tracer is just a more complex
Galton Board.
We hope to use to adapt the approach in [1] to learn a surogate
ray tracer.

## Generating dataset from ray tracer
Todo `@Tiffany`.

## Reproducing ML models and adaptation for ray tracing
Todo `@Hina`.

## Exploring learnable scenes and parameters
### Choice of materials
`refractive` and `diffuse` materials seem to be good choices.
 * Rays incident on `refractive` material break into a transmitted and reflected
   component; this can be simulated with a Bernouilli distribution 
 * Rays incident on `diffuse` materials reflect light _in all directions_, according
   to Lambert's cosine law; this can be simulated with samples from a probability density
   function.

### First results
This is a simple setup with a refractive blue glass ball and diffuse walls
(100x100 image with 1000 samples per pixel and 4 max diffusion ray depth).
We can see that interesting shades appear, although it is surprising how uniform
the refractive glass looks.

<img src="images/210316_1000_depth4.png" width=300>


Adding more lights might make things more interesting.
I will continue to work on different combinations and will aim to have a few options
to choose from, with varying complexity.

### Choice of free parameter
Todo.

## Bibliography
[1] Johann Brehmer, Gilles Louppe, Juan Pavez, and Kyle Cranmer. Mining gold from implicit models to improve likelihood-free inference. PNAS, 2020. [[1805.12244](https://arxiv.org/abs/1805.12244)]
