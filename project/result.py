## class that holds a grid/heightmap, as well as the config for simulation
## this is what we interact with in the jupyter notebook to generate the figures

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import grid
import bunny


@dataclass
class Result:
    def __init__(
        self,
        dK,
        dt,
        max_ns: int | None = None,
        flat_terrain=False,
        grass_consumption: jnp.ndarray | None = None,
        mix_eps=0.0,
        steps_per_frame=20,
        w=320,
        h=320,
        key_n=42,
    ):
        """
        nS: number of bunny species
        """
        self.w = w
        self.h = h
        self.key_n = key_n

        nK = len(dK)
        self.nS = max_ns or nK - 1
        self.actual_nK = self.nS + 1

        self.s0, self.heightmap = grid.reset_grid(
            self.w, self.h, self.actual_nK, key_n=key_n
        )
        self.gc = grass_consumption

        self.flat_terrain = flat_terrain  # for disabling the terrain
        if self.flat_terrain:
            self.heightmap = jnp.zeros((w, h))

        self.gc = (
            jnp.full((self.nS,), 0.95)
            if grass_consumption is None
            else grass_consumption
        )
        self.mix_eps = mix_eps

        self.solver = bunny.bunny_dynamics(
            dK[: self.actual_nK],
            dt,
            feed=0.0367,
            kill=jnp.array([0.0649, 0.05])[: self.nS],
            reproduction=jnp.array([1.0, 0.75])[: self.nS],
            grass_consumption=self.gc,
            terrain=self.heightmap,
            mix_eps=self.mix_eps,
        )

        # step over the solver but not every single frame
        self.stepper = jax.jit(bunny.step_n(self.solver, steps_per_frame))

        self.contours = [] if self.flat_terrain else grid.get_contours(self.heightmap)

    def _fmt_(self):
        a = f"{self.w}x{self.h}_b{self.nS}"
        if self.mix_eps > 0.0:
            a += "_mix"
        if self.flat_terrain:
            a += "_flat"
        return a

    def simulate(self, max_steps=750):
        xs, xs_history = jax.lax.scan(self.stepper, self.s0, length=max_steps)
        return xs, xs_history
