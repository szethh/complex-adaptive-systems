## here are all the actual functions for the dynamics, jax helpers & other stuff

import jax
import jax.numpy as jnp


# these are copy pasted from the last exercise
def rk4(f, dt):
    @jax.jit
    def step(x, t):
        k1 = f(t, x)
        k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = f(t + dt, x + dt * k3)

        xn1 = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return xn1

    return step


# implement diffusion as convolving this laplacian matrix over the K matrix
lapl = jnp.array([[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]])
dx = 2.0
lapl = lapl / (dx * dx)


def diffuse(k_i, dk):
    return dk * jax.scipy.signal.convolve2d(
        k_i, lapl, mode="same", boundary="fill", fillvalue=0
    )


# since scanning over every single timepoint consumes too much memory
# and evaluating every time step is not relevant
# we have a helper to scan and return every n steps
def step_n(solver, n):
    def _step(x, _=None):
        def body(x, _):
            return solver(x, 0.0), None

        x, _ = jax.lax.scan(body, x, None, length=n)
        return x, x

    return _step


# grass "carrying capacity" given a terrain heightmap
# found by playing around a bit in desmos
# https://www.desmos.com/calculator/bkiusgkzow
def capacity_from_height(H, strength=0.3, gamma=2.0):
    return 1.0 - strength * jnp.abs(H) ** gamma


# main solver
def bunny_dynamics(
    dK, dt, feed, kill, reproduction, grass_consumption, terrain, mix_eps=0.0
):
    H = terrain  # (w, h)
    # max carrying capacity of grass at (x, y), given heightmap
    K_G = capacity_from_height(H)

    def f(t, c):
        K = c  # (S+1, w, h)
        G = K[0]  # grass
        B = K[1:]  # bunnies

        # reproduction
        r = reproduction[:, None, None]  # (S, 1, 1)
        R = r * G * B**2 + 0.01 * r * G * B

        # grass consumption
        # grass is determined by a max of K_G
        # and it is reduced by the grass consumption of each bunny species
        # times the amount of new bunnies R
        dG = feed * (K_G - G) - jnp.sum(grass_consumption[:, None, None] * R, axis=0)

        # bunnie increase follows normal reaction diffusion
        dB = R - (kill + feed)[:, None, None] * B

        # we mix a percentage of species to avoid harsh borders
        total_B = jnp.sum(B, axis=0, keepdims=True)
        dB += mix_eps * (total_B - B) * (G > 0.2)

        # diffusion
        k_diffused = jax.vmap(diffuse, in_axes=(0, 0))(K, dK)

        # next_K = k_diffused + jnp.array([dG, dB1, dB2])
        next_K = k_diffused + jnp.concatenate([jnp.array([dG]), dB])

        return next_K

    solver = jax.jit(rk4(f, dt))
    return solver
