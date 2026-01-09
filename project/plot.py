## plotting utilities

from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt

def save_figure(fig, filename: str | Path):
    figure_dir = Path("../project-report/img").resolve()
    f = figure_dir / filename
    print(f"Saving figure output to {f}")
    fig.savefig(f, dpi=300, bbox_inches="tight")


def overlay_terrain_contours(contours, ax, cmap="hot", **kwargs):
    n = len(contours)
    cmap = plt.colormaps.get_cmap(cmap)
    ax.set_prop_cycle('color', [cmap(i) for i in jnp.linspace(0, 1, n)])
    for i in range(n):
        ax.plot(*contours[i].T, **kwargs)

def render_frame(frame, heightmap, cmap=None):
    x = jnp.clip(frame, 0, 1)
    grass = x[0]
    bunnies = x[1:]
    total = bunnies.sum(axis=0) + 1e-6
    # this doesnt look very good
    # density = jnp.tanh(2.0 * total)

    B1 = bunnies[0]
    nS = len(bunnies)
    B2 = bunnies[1] if nS > 1 else jnp.zeros_like(B1)

    if bunnies.shape[0] > 1 or cmap is None:
        R = B1 / total
        G = grass
        B = B2 / total
        rgb = jnp.stack([R, G, B], axis=-1)
    else:
        rgb = cmap(B1)[..., :3]
    # rgb = rgb * density[..., None]

    if heightmap.max() > 1e-3:
        # shade the terrain a bit
        Hn = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        shade = 0.8 + 0.4 * Hn
        rgb = rgb * shade[..., None]

    return (jnp.clip(rgb, 0, 1) * 255).astype(jnp.uint8)