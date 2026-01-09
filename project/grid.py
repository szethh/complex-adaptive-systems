## terrain & grid generation

import jax.numpy as jnp
import jax.random as jr
import numpy as np  # some libs dont like jax numpy arrays
from pathlib import Path
import opensimplex as opsx
import skimage.measure as measure


# generate terrain or load from cache
# turns out opensimplex can be quite slow
def get_heightmap(w, h, seed, scale=24.0, regenerate=False):
    fname = f"heightmap_w{w}_h{h}_seed{seed}_scale{scale}.npy"
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)

    file = out_dir / fname
    if file.exists() and not regenerate:
        return jnp.asarray(np.load(file))

    opsx.seed(seed)
    heightmap = opsx.noise2array(jnp.arange(w) / scale, jnp.arange(h) / scale)
    np.save(file, np.asarray(0.5 * (1.0 + heightmap)))
    return heightmap


# generate a new grid + heightmap
def reset_grid(w, h, nK, key_n=42, seed_size=110):
    key = jr.key(key_n)

    grid = jnp.zeros((nK, w, h)).at[0].set(1.0)

    # (S, n_locations, 2), last dimension corresponds to x,y
    seed_locations = jr.uniform(
        key, shape=(nK - 1, seed_size, 2), minval=0.0, maxval=jnp.array([w, h])
    )
    for i, channel in enumerate(seed_locations):
        for x, y in channel:
            grid = draw_circle(grid, x, y, 2, 0.6, channel=i + 1)
    # grid = draw_circle(grid, w//2-5, h//2-5, 5, .4, channel=1)
    heightmap = get_heightmap(w, h, seed=key_n)

    return grid, heightmap


# get contour lines like in maps
def get_contours(heightmap):
    levels = jnp.linspace(heightmap.min(), heightmap.max(), 8)
    contours = []
    for level in levels:
        cc = measure.find_contours(np.asarray(heightmap), level)
        for c in cc:
            if c.shape[0] >= 2:  # must have at least 2 points
                contours.append(c[:, ::-1])
    return contours


# draws a circle that falls off in intensity. this is used to:
# - seed the initial bunny populations
# - add a new patch in the interactive editor
def draw_circle(grid, cx, cy, radius, strength, channel=1):
    """
    grid: (C, W, H)
    cx, cy: center coordinates
    radius: circle radius
    strength: strength falloff to write inside the circle
    channel: which channel to write to
    """
    _, w, h = grid.shape

    xs = jnp.arange(w)[:, None]
    ys = jnp.arange(h)[None, :]

    # mask = (xs - cx)**2 + (ys - cy)**2 <= radius**2
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    brush = jnp.exp(-dist2 / (2 * radius**2))

    return grid.at[channel].add(strength * brush)

    # return grid.at[channel].set(
    #     jnp.where(mask, value, grid[channel])
    # )
